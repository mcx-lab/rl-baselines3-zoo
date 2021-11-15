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


def plot_signal(ax: plt.Axes, signal: np.ndarray, label: str, **kwargs):
    time = np.arange(len(signal))
    ax.plot(time, signal, label=label, **kwargs)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Signal value")


def iter_video_frames(video_path: Path):
    vidcap = cv2.VideoCapture(str(video_path))
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            yield image


def stitch_videos(in_path1: Path, in_path2: Path, out_path: Path):

    frames1 = iter_video_frames(in_path1)
    frames2 = iter_video_frames(in_path2)

    with imageio.get_writer(out_path, mode="I", fps=30) as writer:
        # Use itertools.zip_longest to iterate sequences of different length
        # Shorter sequence will be padded with fillvalue = None
        for index, (f1, f2) in enumerate(itertools.zip_longest(frames1, frames2, fillvalue=None)):
            # Loop only to the minimum length iterator
            if f1 is None or f2 is None:
                break
            if index % 100 == 99:
                print(f"{index + 1} frames stitched")
            dsize = f1.shape[1], f1.shape[0]
            f2 = cv2.resize(f2, dsize=dsize)
            stitch = cv2.hconcat([f1, f2])
            writer.append_data(stitch)
    print(f"Stitching completed. In total, {index+1} frames were stitched")


def plot_depth_and_filtered(raw_depth: np.ndarray, filtered_depth: np.ndarray, plot_path: Path):
    plt.figure()
    plt.plot(np.arange(raw_depth.shape[1]), raw_depth[0])
    plt.savefig(str(plot_path.parent / "sanity_check.png"))

    rays = [0, 5, 9, 10, 11, 12, 13]
    num_rays = len(rays)
    fig, ax = plt.subplots(num_rays, 1, figsize=(10, 10))
    for i, ray in enumerate(rays):
        plot_signal(ax[i], raw_depth[:, ray], label=f"Raw depth, ray {ray}")
        plot_signal(ax[i], filtered_depth[:, ray], label=f"Filtered depth, ray {ray}")
        ax[i].legend()
    plt.savefig(str(plot_path))
    plt.close(fig)


def plot_depth_video(depth_data: np.ndarray, save_path: Path):

    with imageio.get_writer(str(save_path), mode="I", fps=30) as writer:
        for i in range(depth_data.shape[0]):
            if i % 100 == 99:
                print(f"{i + 1} frames plotted")
            # Plot the current timestep data
            fig = plt.figure()
            ax = fig.add_subplot()
            data = depth_data[i]
            data += 1  # Shifting for better visualisation
            x_space = np.arange(len(data))
            x_space[-4:] += 1  # Leave a gap for plotting of foot rays
            ax.bar(x=x_space, height=data)
            ax.set_ylim((0, 2))
            fig.canvas.draw()  # Generate the image so it can be saved

            # Convert figure to an image
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format="raw")
            io_buf.seek(0)
            img_arr = np.reshape(
                np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
            )
            io_buf.close()

            # Write to video
            writer.append_data(img_arr)
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
        plot_depth_and_filtered(
            raw_depth,
            filtered_depth,
            plot_path / get_data_name("raw_and_filtered_depth", env_modifier, is_observation_modified),
        )

        raw_depth_video_path = plot_path / (get_data_name("raw_depth_video", env_modifier, is_observation_modified) + ".mp4")
        print("Recording depth video")
        plot_depth_video(raw_depth, raw_depth_video_path)
        replay_and_depth_path = plot_path / (
            get_data_name("replay_and_raw_depth", env_modifier, is_observation_modified) + ".mp4"
        )
        print("Stitching replay with depth video")
        replay_path = env_modifier_to_replay_path_map[env_modifier]
        stitch_videos(replay_path, raw_depth_video_path, replay_and_depth_path)


if __name__ == "__main__":
    main()
