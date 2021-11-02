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
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier
from blind_walking.envs.env_modifiers.heightfield import HeightField
from blind_walking.envs.env_modifiers.stairs import Stairs, boxHalfLength, boxHalfWidth
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.sensors import environment_sensors
from enjoy import Logger
from gym.wrappers import Monitor
from scripts.plot_stats import Plotter

import utils.import_envs  # noqa: F401 pytype: disable=import-error


class MultipleTerrain(EnvModifier):
    def __init__(self):
        super().__init__()
        self.hf = HeightField()
        self.stairs = Stairs()

        self.start_x = 5
        # Stairs parameters
        self.step_rise = 0.05
        self.num_steps = 5
        self.stair_gap = 1.5
        self.step_run = 0.3
        self.stair_length = (self.num_steps - 1) * self.step_run * 2 + boxHalfLength * 2 * 2
        # Heightfield parameters
        self.hf_length = 18

    def _generate(self, env):
        start_x = self.start_x
        self.stairs._generate(env, start_x=start_x, num_steps=self.num_steps, step_rise=self.step_rise, step_run=self.step_run)
        start_x += self.stair_length + self.hf_length / 2
        self.hf._generate(env, start_x=start_x, heightPerturbationRange=0.04)

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


def get_img_from_fig(fig, dpi=24):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=dpi)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
    )
    io_buf.close()
    return img_arr


def get_video_save_path(env: Monitor):
    return os.path.join(
        env.directory,
        "{}.video.{}.video{:06}.mp4".format(env.file_prefix, env.file_infix, env.episode_id),
    )


def get_frames_from_video_path(video_path: str):
    vidcap = cv2.VideoCapture(video_path)
    images = []
    success = True
    while success:
        success, image = vidcap.read()
        images.append(image)
    return images


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
    grid_size = (20, 1)
    grid_unit = 0.05
    grid_transform = (0.15, 0)
    num_timesteps = 1000

    if not args.no_hover:
        # Environment parameters
        robot_sensor_list = []
        env_sensor_list = [
            environment_sensors.LocalTerrainViewSensor(grid_size=grid_size, grid_unit=grid_unit, transform=grid_transform),
        ]
        env_randomizer_list = []
        env_modifier_list = [MultipleTerrain()]
        obs_wrapper = obs_array_wrapper.ObservationDictionaryToArrayWrapper

        # Create environment
        env = gym.make(
            "A1GymEnv-v0",
            robot_sensor_list=robot_sensor_list,
            env_sensor_list=env_sensor_list,
            env_randomizer_list=env_randomizer_list,
            env_modifier_list=env_modifier_list,
            obs_wrapper=obs_wrapper,
        )
        if args.record:
            video_folder = os.path.dirname(__file__)
            env = Monitor(env, video_folder, force=True)
            replay_video_path = get_video_save_path(env)
        env.reset()

        # Create logger
        hm_logger = Logger("heightmap")

        # Move robot across terrain and collect heightmap data
        default_orientation = env.robot._GetDefaultInitOrientation()
        default_position = env.robot._GetDefaultInitPosition()
        zero_action = np.zeros(12)
        position = default_position.copy()
        for _ in range(num_timesteps):
            # Update position
            position[0] += dx
            position[1] += dy
            z_pos = env_modifier_list[0].get_z_position(position[0], position[1])
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
        plotter = Plotter(datapath, "hm_single")
        plotter.plot(columns=[0], savedir=dirpath)

        # Generate GIF of heightmap over time
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
        for i in range(num_timesteps):
            ax.scatter(xx, yy, c=plotter.data[i], vmin=0, vmax=0.3)
            image = get_img_from_fig(fig, dpi=dpi)
            images.append(image)
        plt.close(fig)

        print("Generated images for video")
        # build gif
        files = glob.glob(os.path.join(dirpath, "hm*.png"))
        files = [f for f in files if "_" not in os.path.basename(f)]
        heightmap_video_path = os.path.join(dirpath, "hm.mp4")
        with imageio.get_writer(heightmap_video_path, mode="I", fps=5) as writer:
            for image in images:
                writer.append_data(image)
        print("Created heightmap video")

        # stitch both videos together
        if args.record:
            print(f"Stitching video from {replay_video_path}")

            replay_frames = get_frames_from_video_path(replay_video_path)[:1000]
            heightmap_frames = get_frames_from_video_path(heightmap_video_path)[:1000]

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
