"""Script for plotting depth sensor data, raw and filtered."""

import argparse
import itertools
import json
import tempfile
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from blind_walking.robots.action_filter import ActionFilterButter
from scipy.fft import fft, fftfreq

from utils.visualization import VideoPlotter, stitch_videos


def plot_signal(ax: plt.Axes, signal: np.ndarray, label: str, **kwargs):
    time = np.arange(len(signal))
    ax.plot(time, signal, label=label, **kwargs)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Signal value")


def plot_fft(ax: plt.Axes, signal: np.ndarray, label: str, **kwargs):
    assert len(signal.shape) == 1
    T = 0.033  # env_time_step
    N = len(signal)
    yfreq = fft(signal)
    xfreq = fftfreq(N, T)[: N // 2]
    ax.plot(xfreq, 2.0 / N * np.abs(yfreq[0 : N // 2]), label=label, **kwargs)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Frequency Strength")


def load_data(stats_path: Path, data_prefix: str, sensor_name: str, env_modifier: str, is_observation_modified: bool):
    return np.load(stats_path / f"{get_data_name(data_prefix, sensor_name, env_modifier, is_observation_modified)}.npy")


def get_data_name(data_prefix: str, sensor_name: str, env_modifier: str, is_observation_modified: bool):
    modify_suffix = "modified" if is_observation_modified else "unmodified"
    return f"{data_prefix}_{sensor_name}_{env_modifier}_{modify_suffix}"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", help="Directory in which statistics are saved", type=str)
    parser.add_argument(
        "--terrain-modifiers",
        type=str,
        nargs="+",
        help="Terrain modifiers to examine",
        default=["heightfield"],
    )
    parser.add_argument("--modification-flags", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--plot-path", type=str, help="Directory to save plots", default=None)
    args = parser.parse_args()

    # Create plot directory
    stats_path = Path(args.stats_path)
    plot_path = args.plot_path
    if plot_path is None:
        plot_path = stats_path.parent / "plots"
    else:
        plot_path = Path(args.plot_path)
    plot_path.mkdir(exist_ok=True, parents=True)

    # Plot all data
    env_modifiers = args.terrain_modifiers
    is_observation_modifieds = args.modification_flags
    sensor_names = [f"{w}_flatten" for w in ["depthfr", "depthfl", "depthrl", "depthrr", "depthmiddle"]]
    grid_sizes = [(3, 3), (3, 3), (3, 3), (3, 3), (10, 1)]
    grid_angles = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.1, 0)]
    datalim = (0.2, 1.4)

    for settings in itertools.product(env_modifiers, is_observation_modifieds):

        env_modifier, is_observation_modified = settings
        modification_suffix = "modified" if is_observation_modified else "unmodified"
        with open((stats_path.parent / f"env_modifier_to_replay_path_map_{modification_suffix}.json"), "r") as jsonfile:
            env_modifier_to_replay_path_map = json.load(jsonfile)

        raw_depths = []
        filtered_depths = []
        for sensor_name, grid_size, grid_angle in zip(sensor_names, grid_sizes, grid_angles):
            raw_depth = load_data(stats_path, "raw", sensor_name, env_modifier, is_observation_modified)
            filtered_depth = load_data(stats_path, "filtered", sensor_name, env_modifier, is_observation_modified)
            raw_depths.append(raw_depth)
            filtered_depths.append(filtered_depth)
        raw_depths = np.concatenate(raw_depths, axis=1)
        filtered_depths = np.concatenate(filtered_depths, axis=1)

        barplotter = BarPlot3DVideoPlotter(grid_sizes, sensor_names, datalim)

        # Sanity check the first frame
        fig = plt.figure()
        barplotter.plot_frame(fig, raw_depths[0])
        plt.savefig(str(plot_path / "sanity_check.png"))
        plt.close(fig)

        # Plot depth video similar to hover_robot.py
        depth_video_path = plot_path / (get_data_name("depth", "all", env_modifier, is_observation_modified) + ".mp4")
        replay_path = env_modifier_to_replay_path_map[env_modifier]
        barplotter.plot_video(raw_depths, depth_video_path, verbose=1)
        stitch_path = plot_path / (get_data_name("replay_and_depth", "all", env_modifier, is_observation_modified) + ".mp4")
        stitch_videos(replay_path, depth_video_path, stitch_path, verbose=1)

        # Front foot vs rear foot analysis
        rays = [0, 7, 9, 16, 18, 25, 27, 35, 37, 43]  # front feet  # rear feet  # middle rays
        num_rays = len(rays)
        fig, ax = plt.subplots(num_rays, 1, figsize=(20, 10))
        for i, ray in enumerate(rays):
            plot_fft(ax[i], raw_depths[:, ray], label=f"Raw depth, ray_id {ray}")
            plot_fft(ax[i], filtered_depths[:, ray], label=f"Filtered depth, ray_id {ray}")
            ax[i].legend()
        plt.savefig(str(plot_path / "raw_depth_fft.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
