"""Script for building a hovering robot and collecting heightmap data."""

import argparse
import glob
import io
import os
import cv2
import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import Monitor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier
from blind_walking.envs.env_modifiers.heightfield import HeightField
from blind_walking.envs.env_modifiers.stairs import Stairs, boxHalfLength, boxHalfWidth
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.sensors import environment_sensors
from blind_walking.envs.tasks.forward_task import ForwardTask
from enjoy import Logger
from scripts.plot_stats import Plotter, get_img_from_fig, get_frames_from_video_path, alphanum_key
import utils.import_envs  # noqa: F401 pytype: disable=import-error


class MultipleTerrain(EnvModifier):
    def __init__(self):
        super().__init__()
        self.hf = HeightField()
        self.stairs = Stairs()

        self.start_x = 5
        # Stairs parameters
        self.step_rise = 0.05
        self.num_steps = 10
        self.stair_gap = 1.5
        self.step_run = 0.3
        self.stair_length = (self.num_steps - 1) * self.step_run * 2 + boxHalfLength * 2 * 2
        # Heightfield parameters
        self.hf_length = 18

    def _generate(self, env):
        start_x = self.start_x
        self.stairs._generate(env, start_x=start_x, num_steps=self.num_steps, step_rise=self.step_rise, step_run=self.step_run)
        start_x += self.stair_length + self.hf_length / 2
        self.hf._generate(env, start_x=start_x, heightPerturbationRange=0.08)

    def get_z_position(self, x, y):
        """Get z position for hovering robot at the xy-coord."""
        """NOTE: Position checking are currently hardcoded"""
        if x > self.start_x and x < self.start_x + self.stair_length:
            # On the stairs
            x_stairs = x - self.start_x
            if x_stairs < self.step_run * (self.num_steps - 1):
                # On the ascending stairs
                step_on = 1 + x_stairs // self.step_run
            elif x_stairs < self.step_run * (self.num_steps - 1) + boxHalfLength * 2 * 2:
                # On the top of the stairs
                step_on = self.num_steps
            else:
                # On the descending stairs
                x_stairs -= self.step_run * (self.num_steps - 1) + boxHalfLength * 2 * 2
                step_on = self.num_steps - 1 - x_stairs // self.step_run
            z_pos = step_on * self.step_rise
        elif x > self.start_x + self.stair_length and x < self.start_x + self.stair_length + self.hf_length:
            # On uneven terrain
            z_pos = 0.04
        else:
            z_pos = 0
        return z_pos


def get_video_save_path(env: Monitor):
    return os.path.join(
        env.directory,
        "{}.video.{}.video{:06}.mp4".format(env.file_prefix, env.file_infix, env.episode_id),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-hover", action="store_true", default=False, help="Do not generate heightmap data with hover robot"
    )
    parser.add_argument("--no-plot", action="store_true", default=False, help="Do not generate heightmap plots")
    parser.add_argument("--record", action="store_true", default=False, help="Record video")
    args = parser.parse_args()

    # Data parameters
    dx = 0.05
    dy = 0
    grid_size = (10, 1)
    grid_unit = 0.05  # Unused in LocalTerrainDepthByAngleSensor
    grid_angle = 0.1
    grid_transform = (-0.7, 0)
    num_timesteps = 1000

    if not args.no_hover:
        # Environment parameters
        robot_sensor_list = []
        env_sensor_list = [
            environment_sensors.LocalTerrainDepthByAngleSensor(
                grid_size=grid_size, grid_angle=grid_angle, transform_angle=grid_transform, noisy_reading=False
            ),
        ]
        env_randomizer_list = []
        env_modifier_list = [MultipleTerrain()]
        obs_wrapper = obs_array_wrapper.ObservationDictionaryToArrayWrapper
        task = ForwardTask()

        # Create environment
        env = gym.make(
            "A1GymEnv-v0",
            robot_sensor_list=robot_sensor_list,
            env_sensor_list=env_sensor_list,
            env_randomizer_list=env_randomizer_list,
            env_modifier_list=env_modifier_list,
            obs_wrapper=obs_wrapper,
            task=task,
        )
        if args.record:
            video_folder = os.path.dirname(__file__)
            env = Monitor(env, video_folder, force=True)
            replay_video_path = get_video_save_path(env)
        env.reset()

        # Create logger
        hm_logger = Logger("heightmap")

        # Move robot across terrain and collect heightmap data
        default_orientation = env.pybullet_client.getQuaternionFromEuler([0, 0, 0])
        default_position = env.robot._GetDefaultInitPosition()
        zero_action = np.zeros(12)
        position = default_position.copy()
        for _ in range(num_timesteps):
            # Update position
            position[0] += dx
            position[1] += dy
            # Calculate z pos 5 timestep faster to avoid legs hitting the stairs
            z_pos = env_modifier_list[0].get_z_position(position[0] + 5 * dx, position[1] + 5 * dy)
            position[2] = default_position[2] + z_pos
            env.pybullet_client.resetBasePositionAndOrientation(env.robot.quadruped, position, default_orientation)
            obs, _, _, _ = env.step(zero_action)
            # Record heightmap data
            hm_logger.update([obs.reshape(grid_size)])

        # Close environment and save data
        env.close()
        hm_logger.save(os.path.dirname(__file__))
        print("Collected data")

    if not args.no_plot:
        dirpath = os.path.dirname(__file__)
        datapath = os.path.join(dirpath, "heightmap.npy")
        data = np.load(datapath)

        # Plot one data point of the heightmap
        for i in range(grid_size[0]):
            plotter = Plotter(datapath, f"hm_single{i}")
            plotter.plot(columns=[i], ylim=(0.2, 1.4), savedir=dirpath)

        # Generate GIF of heightmap over time
        if grid_size[0] == 1 or grid_size[1] == 1:
            # bar graph plot
            for i in range(num_timesteps):
                plt.figure()
                data = np.array(plotter.data[i])[:, 0]
                plt.bar(x=np.arange(len(data)), height=np.flip(data))
                plt.ylim((0.2, 1.4))
                plt.savefig(os.path.join(dirpath, f"tmp{i}"))
                plt.close()
            print("Generated images for video")
            # build gif
            files = glob.glob(os.path.join(dirpath, "tmp*.png"))
            files.sort(key=alphanum_key)
            heightmap_video_path = os.path.join(dirpath, "hm.mp4")
            with imageio.get_writer(heightmap_video_path, mode="I", fps=30) as writer:
                for f in files:
                    image = imageio.imread(f)
                    writer.append_data(image)
            print("Created heightmap video")
            # remove images
            for f in files:
                os.remove(f)
            print("Removed unnessary image files")
        else:
            # scatter plot
            kx = grid_size[0] / 2 - 0.5
            xvalues = np.linspace(-kx * grid_unit, kx * grid_unit, num=grid_size[0])
            ky = grid_size[1] / 2 - 0.5
            yvalues = np.linspace(-ky * grid_unit, ky * grid_unit, num=grid_size[1])
            xx, yy = np.meshgrid(xvalues, yvalues)
            # generate images
            images = []
            dpi = 60
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            for i in range(num_timesteps):
                img = ax.scatter(xx, yy, c=plotter.data[i], vmin=0.2, vmax=0.8)
                fig.colorbar(img, cax=cax, orientation="vertical")
                image = get_img_from_fig(fig, dpi=dpi)
                images.append(image)
            plt.close(fig)
            print("Generated images for video")
            # build gif
            heightmap_video_path = os.path.join(dirpath, "hm.mp4")
            with imageio.get_writer(heightmap_video_path, mode="I", fps=30) as writer:
                for image in images:
                    writer.append_data(image)
            print("Created heightmap video")

        # stitch both videos together
        if args.record:
            print(f"Stitching video from {replay_video_path}")
            replay_frames = get_frames_from_video_path(replay_video_path)[:num_timesteps]
            heightmap_frames = get_frames_from_video_path(heightmap_video_path)[:num_timesteps]

            assert len(replay_frames) == len(heightmap_frames)
            replay_and_heightmap_frames = []
            for rp, hm in zip(replay_frames, heightmap_frames):
                dsize = rp.shape[1], rp.shape[0]
                hm = cv2.resize(hm, dsize=dsize)
                rp_and_hm = cv2.hconcat([rp, hm])
                replay_and_heightmap_frames.append(rp_and_hm)

            stitch_video_path = os.path.join(dirpath, "replay_and_hm.mp4")
            with imageio.get_writer(stitch_video_path, mode="I", fps=30) as writer:
                for rp_and_hm in replay_and_heightmap_frames:
                    writer.append_data(rp_and_hm)
            print("Finished stitching videos")


if __name__ == "__main__":
    main()
