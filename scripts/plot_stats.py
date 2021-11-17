import argparse
import os
import io
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import cv2
import itertools
from typing import Union
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable


def tryint(s):
    try:
        return int(s)
    except Exception as _:
        return s


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


class Plotter:
    def __init__(self, datapath: str = None, name: str = "plot"):
        self.data = np.load(datapath)
        self.name = name

    def plot(self, columns=None, ylim=None, savedir: str = None):
        plt.figure()
        # timesteps as x-axis
        t = np.arange(self.data.shape[0])
        columns = columns if len(columns) else np.arange(self.data.shape[1])
        for i in columns:
            plt.plot(t, self.data[:, i])
        if ylim:
            plt.ylim(ylim)
        plt.savefig(os.path.join(savedir, self.name))
        plt.close()


def get_img_from_fig(fig, dpi=24):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=dpi)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
    )
    io_buf.close()
    return img_arr


def get_frames_from_video_path(video_path: str):
    vidcap = cv2.VideoCapture(video_path)
    images = []
    success = True
    while success:
        success, image = vidcap.read()
        images.append(image)
    return images


def iter_video_frames(video_path: Union[Path, str]):
    """Iterate over frames in a video.

    Avoids loading all frames into memory at once to prevent OoM errors."""
    vidcap = cv2.VideoCapture(str(video_path))
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            yield image


def stitch_videos(in_path1: Path, in_path2: Union[Path, str], out_path: Union[Path, str], verbose: int = 0):
    """Stitch two videos together by concatenating horizontally.

    Stitched video will be the length of the shorter video.

    Args:
        - in_path1: Path to first video
        - in_path2: Path to second video. Will be resized to size of first video
        - out_path: Path to save output video
        - verbose: Set verbose = 1 to print stitching information.
    """
    frames1 = iter_video_frames(in_path1)
    frames2 = iter_video_frames(in_path2)

    with imageio.get_writer(out_path, mode="I", fps=30) as writer:
        # Use itertools.zip_longest to iterate sequences of different length
        # Shorter sequence will be padded with fillvalue = None
        for index, (f1, f2) in enumerate(itertools.zip_longest(frames1, frames2, fillvalue=None)):
            # Loop only to the minimum length iterator
            if f1 is None or f2 is None:
                break
            if verbose >= 1 and index % 100 == 99:
                print(f"{index + 1} frames stitched")
            dsize = f1.shape[1], f1.shape[0]
            f2 = cv2.resize(f2, dsize=dsize)
            stitch = cv2.hconcat([f1, f2])
            writer.append_data(stitch)
    if verbose >= 1:
        print(f"Stitching completed. In total, {index+1} frames were stitched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-folder", help="input path to folder which holds the stats data", type=str, default="./")
    parser.add_argument(
        "-s", "--stitch-path", help="input path to video which is to be stitched together", type=str, default=None
    )
    args = parser.parse_args()

    data_name = "true_extrinsics"
    files = glob.glob(os.path.join(args.input_folder, f"{data_name}*.npy"))
    for f in files:
        plotter = Plotter(f, os.path.splitext(os.path.basename(f))[0])
        plotter.plot(ylim=(-10, 10), savedir=args.input_folder)

    data_name = "observations"
    files = glob.glob(os.path.join(args.input_folder, f"{data_name}*.npy"))
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        dirpath = args.input_folder

        # Plot one data point of the heightmap
        plotter = Plotter(f, basename + "_hmsingle")
        plotter.plot(columns=[-1], savedir=dirpath)

        num_obs = len(plotter.data[0])
        num_timesteps = len(plotter.data)
        hmobs_startindex = 46
        grid_sizes = [(3, 3), (3, 3), (3, 3), (3, 3), (10, 1)]
        grid_angles = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.1, 0)]
        grid_transforms = [(-0.6, -0.4), (-0.6, 0.4), (0.4, -0.4), (0.4, 0.4), (-0.8, 0)]
        grid_names = ["depthfr", "depthfl", "depthrr", "depthrl", "depthmiddle"]

        datalim = (0, 6)
        datashift = 3  # Amount to shift for better visualisation
        plotter.data = plotter.data + datashift
        # Generate GIF of heightmap over time
        if len(grid_sizes) == 1 and (grid_sizes[0][0] == 1 or grid_sizes[0][1] == 1):
            # bar graph plot
            for i in range(num_timesteps):
                plt.figure()
                data = plotter.data[i][hmobs_startindex:]
                plt.bar(x=np.arange(len(data)), height=data)
                plt.ylim(datalim)
                plt.savefig(os.path.join(dirpath, f"tmp{i}"))
                plt.close()
            print("Generated images for video")
            # build gif
            files = glob.glob(os.path.join(dirpath, "tmp*.png"))
            files.sort(key=alphanum_key)
            heightmap_video_path = os.path.join(dirpath, basename + "_hm.mp4")
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
            # 3d bar plot
            grid_end_indices = [np.prod(s) for s in grid_sizes]
            grid_end_indices = np.cumsum(grid_end_indices) + hmobs_startindex
            subplot_size = "2" + str(int(np.ceil(len(grid_sizes) / 2)))
            for t in range(num_timesteps):
                fig = plt.figure()
                for i in range(len(grid_sizes)):
                    ax = fig.add_subplot(int(subplot_size + str(i + 1)), projection="3d")
                    x = np.arange(grid_sizes[i][0])
                    y = np.arange(grid_sizes[i][1])
                    xx, yy = np.meshgrid(x, y)
                    x, y = xx.ravel(), yy.ravel()
                    z = np.zeros(len(x))
                    dx = dy = 1
                    start_index = hmobs_startindex if i == 0 else grid_end_indices[i - 1]
                    dz = plotter.data[t][start_index : grid_end_indices[i]]
                    ax.set_zlim(datalim)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    # ax.set_zticklabels([])
                    ax.bar3d(x, y, z, dx, dy, dz, shade=True)
                    ax.set_title(grid_names[i])
                plt.savefig(os.path.join(dirpath, f"tmp{t}"))
                plt.close()
            print("Generated images for video")
            # build gif
            files = glob.glob(os.path.join(dirpath, "tmp*.png"))
            files.sort(key=alphanum_key)
            heightmap_video_path = os.path.join(dirpath, basename + "_hm.mp4")
            with imageio.get_writer(heightmap_video_path, mode="I", fps=30) as writer:
                for f in files:
                    image = imageio.imread(f)
                    writer.append_data(image)
            print("Created heightmap video")
            # remove images
            for f in files:
                os.remove(f)
            print("Removed unnessary image files")

        if args.stitch_path:
            replay_video_path = args.stitch_path
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
