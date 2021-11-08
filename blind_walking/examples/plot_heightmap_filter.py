"""Script for plotting action filter."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from blind_walking.robots.action_filter import ActionFilterButter


def plot_signal(signal: np.ndarray, label: str):
    time = np.arange(len(signal))
    plt.plot(time, signal, label=label)
    plt.xlabel("Time step")
    plt.ylabel("Signal value")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath", type=str, help="Path to the heightmap data", default="blind_walking/examples/heightmap.npy"
    )
    parser.add_argument("--savedir", type=str, help="Directory to save plots", default="plots")
    args = parser.parse_args()

    heightmap_data = np.load(args.datapath)[:, :, 0]  # (1000, num_rays)
    filter = ActionFilterButter(lowcut=[0], highcut=[10.0], order=1, sampling_rate=100, num_joints=20)

    filtered_data = np.zeros_like(heightmap_data)
    for t in range(heightmap_data.shape[0]):
        filtered_data[t] = filter.filter(heightmap_data[t])

    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    rays = [0, 10, 19]
    plt.figure()
    for ray in rays:
        plot_signal(heightmap_data[:, ray], label=f"Raw depth, ray {ray}")
        plot_signal(filtered_data[:, ray], label=f"Filtered depth, ray {ray}")
    plt.legend()
    plt.savefig(str(savedir / "depth_and_filtered.png"))


if __name__ == "__main__":
    main()
