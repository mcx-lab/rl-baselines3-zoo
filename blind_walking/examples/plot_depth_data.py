"""Script for plotting depth sensor data, raw and filtered."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from blind_walking.robots.action_filter import ActionFilterButter


def plot_signal(ax: plt.Axes, signal: np.ndarray, label: str, **kwargs):
    time = np.arange(len(signal))
    ax.plot(time, signal, label=label, **kwargs)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Signal value")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-depth-path", type=str, help="Path to the raw depth sensor data")
    parser.add_argument("--filtered-depth-path", type=str, help="Path to the filtered depth data")
    parser.add_argument("--save-path", type=str, help="Directory to save plots", default="plots/depth_and_filtered.png")
    args = parser.parse_args()

    raw_depth = np.load(args.raw_depth_path)  # (episode_len, num_rays)
    filtered_depth = np.load(args.filtered_depth_path)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)

    plt.figure()
    plt.plot(np.arange(raw_depth.shape[1]), raw_depth[0])
    plt.savefig(str(save_path.parent / "sanity_check.png"))

    rays = [0, 10, 19]
    num_rays = len(rays)
    fig, ax = plt.subplots(num_rays, 1, figsize=(10, 10))
    for i, ray in enumerate(rays):
        plot_signal(ax[i], raw_depth[:, ray], label=f"Raw depth, ray {ray}")
        plot_signal(ax[i], filtered_depth[:, ray], label=f"Filtered depth, ray {ray}")
        ax[i].legend()
    plt.savefig(str(save_path))


if __name__ == "__main__":
    main()
