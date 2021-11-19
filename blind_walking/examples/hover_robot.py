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
from blind_walking.envs.sensors import robot_sensors
from blind_walking.envs.tasks.forward_task import ForwardTask
from enjoy import Logger
from gym.wrappers import Monitor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scripts.plot_stats import Plotter, alphanum_key, get_frames_from_video_path, get_img_from_fig

import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils.visualization import VideoPlotter, stitch_videos


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


class BarPlot3DVideoPlotter(VideoPlotter):
    # Create video of 3D bar plots for depth-by-angle sensor
    def __init__(self, grid_sizes, grid_names, datalim):
        self.grid_sizes = grid_sizes
        self.grid_names = grid_names
        self.datalim = datalim
        self.grid_end_indices = np.cumsum([np.prod(s) for s in grid_sizes])
        self.subplot_size = "2" + str(int(np.ceil(len(grid_sizes) / 2)))

    def plot_frame(self, fig: plt.Figure, data: np.ndarray) -> plt.Figure:
        for i in range(len(self.grid_sizes)):
            ax = fig.add_subplot(int(self.subplot_size + str(i + 1)), projection="3d")
            x = np.arange(self.grid_sizes[i][0])
            y = np.arange(self.grid_sizes[i][1])
            xx, yy = np.meshgrid(x, y)
            x, y = xx.ravel(), yy.ravel()
            z = np.zeros(len(x))
            dx = dy = 1
            start_index = 0 if i == 0 else self.grid_end_indices[i - 1]
            dz = data[start_index : self.grid_end_indices[i]]
            ax.set_zlim(self.datalim)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # ax.set_zticklabels([])
            ax.bar3d(x, y, z, dx, dy, dz, shade=True)
            ax.set_title(self.grid_names[i])
        return fig


def main():  # noqa: C901
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
    grid_sizes = [(3, 3), (3, 3), (3, 3), (3, 3), (10, 1)]
    grid_angles = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.1, 0)]
    grid_transforms = [(-0.6, -0.4), (-0.6, 0.4), (0.4, -0.4), (0.4, 0.4), (-0.8, 0)]
    grid_names = ["depthfr", "depthfl", "depthrr", "depthrl", "depthmiddle"]
    num_timesteps = 800

    if not args.no_hover:
        # Environment parameters
        robot_sensor_list = [
            robot_sensors.LocalTerrainDepthByAngleSensor(
                grid_size=grid_sizes[i],
                grid_angle=grid_angles[i],
                transform_angle=grid_transforms[i],
                noisy_reading=False,
                use_filter=True,
                name=grid_names[i],
            )
            for i in range(len(grid_sizes))
        ]
        env_sensor_list = []
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
            hm_logger.update([obs])

        # Close environment and save data
        env.close()
        hm_logger.save(os.path.dirname(__file__))
        print("Collected data")

    if not args.no_plot:
        datalim = (0.2, 1.4)
        dirpath = os.path.dirname(__file__)
        datapath = os.path.join(dirpath, "heightmap.npy")
        data = np.load(datapath)

        # Plot one data point of the heightmap
        plotter = Plotter(datapath, "hm_single")
        plotter.plot(columns=[-1], ylim=datalim, savedir=dirpath)

        # Generate GIF of heightmap over time
        # # 3d bar plot
        heightmap_plotter = BarPlot3DVideoPlotter(grid_sizes=grid_sizes, grid_names=grid_names, datalim=datalim)
        heightmap_video_path = os.path.join(dirpath, "hm.mp4")
        heightmap_plotter.plot_video(plotter.data, heightmap_video_path, verbose=1)
        # stitch both videos together
        if args.record:
            print(f"Stitching video from {replay_video_path} with {heightmap_video_path}")
            stitch_video_path = os.path.join(dirpath, "replay_and_hm.mp4")
            stitch_videos(replay_video_path, heightmap_video_path, stitch_video_path, verbose=1)
            print("Finished stitching videos")


if __name__ == "__main__":
    main()
