"""Script for plotting depth sensor data, raw and filtered."""

import argparse
import io
import itertools
import json
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


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


def plot_depth_and_filtered_fft(raw_depth: np.ndarray, filtered_depth: np.ndarray, plot_path: Path):
    plt.figure()
    plt.plot(np.arange(raw_depth.shape[1]), raw_depth[0])
    plt.savefig(str(plot_path.parent / "sanity_check.png"))

    rays = np.arange(14)
    num_rays = len(rays)
    fig, ax = plt.subplots(num_rays, 1, figsize=(20, 10))
    for i, ray in enumerate(rays):
        plot_fft(ax[i], raw_depth[:, ray], label=f"Raw depth, ray {ray}")
        plot_fft(ax[i], filtered_depth[:, ray], label=f"Filtered depth, ray {ray}")
        ax[i].legend()
    plt.savefig(str(plot_path))
    plt.close(fig)


def load_data(stats_path: Path, data_prefix: str, env_modifier: str, is_observation_modified: bool):
    return np.load(stats_path / f"{get_data_name(data_prefix, env_modifier, is_observation_modified)}.npy")


def get_data_name(data_prefix: str, env_modifier: str, is_observation_modified: bool):
    modify_suffix = "modified" if is_observation_modified else "unmodified"
    return f"{data_prefix}_{env_modifier}_{modify_suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", help="Directory in which statistics are saved", type=str)
    parser.add_argument(
        "--terrain-modifiers",
        type=str,
        nargs="+",
        help="Terrain modifiers to examine",
        default=["heightfield", "stairs_0", "stairs_1"],
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

    for settings in itertools.product(env_modifiers, is_observation_modifieds):
        env_modifier, is_observation_modified = settings
        modification_suffix = "modified" if is_observation_modified else "unmodified"
        with open((stats_path.parent / f"{modification_suffix}_env_modifier_to_replay_path_map.json"), "r") as jsonfile:
            env_modifier_to_replay_path_map = json.load(jsonfile)

        print(f"Creating plots for env_modifier={env_modifier}, is_observation_modified={is_observation_modified}")
        raw_depth = load_data(stats_path, "raw_depth", env_modifier, is_observation_modified)
        filtered_depth = load_data(stats_path, "filtered_depth", env_modifier, is_observation_modified)

        print("Plotting depth and filtered")
        plot_depth_and_filtered_fft(
            raw_depth,
            filtered_depth,
            plot_path / get_data_name("raw_and_filtered_depth_fft", env_modifier, is_observation_modified),
        )


if __name__ == "__main__":
    main()
