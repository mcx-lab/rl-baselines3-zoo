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


def plot_cumulative_fft(ax: plt.Axes, signal: np.ndarray, label: str, **kwargs):
    assert len(signal.shape) == 1
    T = 0.033  # env_time_step
    N = len(signal)
    yfreq = fft(signal)
    xfreq = fftfreq(N, T)[: N // 2]
    ax.plot(xfreq, np.cumsum(2.0 / N * np.abs(yfreq[0 : N // 2])), label=label, **kwargs)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cumulative Frequency Strength")


def compare_fft(depth1: np.ndarray, depth2: np.ndarray, plot_path: Path):
    plt.figure()

    rays = np.arange(14)
    num_rays = len(rays)
    fig, ax = plt.subplots(num_rays, 1, figsize=(20, 10))
    for i, ray in enumerate(rays):
        plot_cumulative_fft(ax[i], depth1[:, ray], label=f"Depth1, ray {ray}")
        plot_cumulative_fft(ax[i], depth2[:, ray], label=f"Depth2, ray {ray}")
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
    parser.add_argument("--stats-path-1", help="Directory in which statistics are saved", type=str)
    parser.add_argument("--stats-path-2", help="Directory in which statistics are saved", type=str)
    parser.add_argument(
        "--terrain-modifiers",
        type=str,
        nargs="+",
        help="Terrain modifiers to examine",
        default=["heightfield", "stairs_0", "stairs_1"],
    )
    parser.add_argument("--modification-flags", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    # Create plot directory
    stats_path_1 = Path(args.stats_path_1)
    stats_path_2 = Path(args.stats_path_2)
    plot_path_1 = stats_path_1.parent / "plots"
    plot_path_2 = stats_path_2.parent / "plots"
    plot_path_1.mkdir(exist_ok=True, parents=True)
    plot_path_2.mkdir(exist_ok=True, parents=True)

    # Plot all data
    env_modifiers = args.terrain_modifiers
    is_observation_modifieds = args.modification_flags

    for settings in itertools.product(env_modifiers, is_observation_modifieds):
        env_modifier, is_observation_modified = settings
        modification_suffix = "modified" if is_observation_modified else "unmodified"
        with open((stats_path_1.parent / f"{modification_suffix}_env_modifier_to_replay_path_map.json"), "r") as jsonfile:
            env_modifier_to_replay_path_map_1 = json.load(jsonfile)
        with open((stats_path_2.parent / f"{modification_suffix}_env_modifier_to_replay_path_map.json"), "r") as jsonfile:
            env_modifier_to_replay_path_map_2 = json.load(jsonfile)

        print(f"Creating plots for env_modifier={env_modifier}, is_observation_modified={is_observation_modified}")
        depth1 = load_data(stats_path_1, "raw_depth", env_modifier, is_observation_modified)
        depth2 = load_data(stats_path_2, "raw_depth", env_modifier, is_observation_modified)

        compare_fft(
            depth1,
            depth2,
            plot_path_1 / get_data_name("depth_compare", env_modifier, is_observation_modified),
        )


if __name__ == "__main__":
    main()
