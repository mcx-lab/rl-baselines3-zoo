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


def plot_signal(ax: plt.Axes, signal: np.ndarray, label: str, **kwargs):
    time = np.arange(len(signal))
    ax.plot(time, signal, label=label, **kwargs)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Signal value")


def get_frames_from_video_path(video_path: str):
    vidcap = cv2.VideoCapture(video_path)
    images = []
    success = True
    while success:
        success, image = vidcap.read()
        # success will be False on last frame
        if success:
            images.append(image)
    return images


def stitch_videos(in_path1, in_path2, out_path):

    frames1 = get_frames_from_video_path(in_path1)
    frames2 = get_frames_from_video_path(in_path2)

    minlen = min(len(frames1), len(frames2))
    frames1 = frames1[:minlen]
    frames2 = frames2[:minlen]

    stitch_frames = []
    for f1, f2 in zip(frames1, frames2):
        dsize = f1.shape[1], f1.shape[0]
        f2 = cv2.resize(f2, dsize=dsize)
        stitch = cv2.hconcat([f1, f2])
        stitch_frames.append(stitch)

    with imageio.get_writer(out_path, mode="I", fps=30) as writer:
        for stitch in stitch_frames:
            writer.append_data(stitch)


def plot_depth_and_filtered(raw_depth: np.ndarray, filtered_depth: np.ndarray, plot_path: Path):
    plt.figure()
    plt.plot(np.arange(raw_depth.shape[1]), raw_depth[0])
    plt.savefig(str(plot_path.parent / "sanity_check.png"))

    rays = [0, 5, 9]
    num_rays = len(rays)
    fig, ax = plt.subplots(num_rays, 1, figsize=(10, 10))
    for i, ray in enumerate(rays):
        plot_signal(ax[i], raw_depth[:, ray], label=f"Raw depth, ray {ray}")
        plot_signal(ax[i], filtered_depth[:, ray], label=f"Filtered depth, ray {ray}")
        ax[i].legend()
    plt.savefig(str(plot_path))


def plot_depth_video(depth_data: np.ndarray, save_path: Path):

    tempdir = tempfile.TemporaryDirectory()
    tempdir_path = Path(tempfile.name)

    filenames = []
    for i in range(depth_data.shape[0]):
        filename = str(tempdir_path / f"tmp{i}.png")
        filenames.append(filename)
        plt.figure()
        data = depth_data[i]
        data += 1  # Shifting for better visualisation
        x_space = np.arange(len(data))
        x_space[-4:] += 1  # Leave a gap for plotting of foot rays
        plt.bar(x=x_space, height=data)
        plt.ylim((0, 2))
        plt.savefig(filename)
        plt.close()
    print("Generated images for video")

    with imageio.get_writer(str(save_path), mode="I", fps=30) as writer:
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)
    print("Created heightmap video")

    # remove images
    tempdir.cleanup()
    print("Removed unnessary image files")


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

    with open((stats_path.parent / "env_modifier_to_replay_path_map.json", "r")) as jsonfile:
        env_modifier_to_replay_path_map = json.load(jsonfile)

    for settings in itertools.product(env_modifiers, is_observation_modifieds):
        env_modifier, is_observation_modified = settings
        raw_depth = load_data(stats_path, "raw_depth", env_modifier, is_observation_modified)
        filtered_depth = load_data(stats_path, "filtered_depth", env_modifier, is_observation_modified)
        plot_depth_and_filtered(
            raw_depth,
            filtered_depth,
            plot_path / get_data_name("raw_and_filtered_depth", env_modifier, is_observation_modified),
        )

        raw_depth_video_path = plot_path / (get_data_name("raw_depth_video", env_modifier, is_observation_modified + ".mp4"))
        plot_depth_video(raw_depth, raw_depth_video_path)
        stitch_videos(env_modifier_to_replay_path_map[env_modifier], raw_depth_video_path)


if __name__ == "__main__":
    main()
