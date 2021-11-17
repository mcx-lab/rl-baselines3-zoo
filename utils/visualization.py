import cv2
import itertools
import io
import imageio
from pathlib import Path
from typing import Union, Iterable
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC


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


class VideoPlotter(ABC):
    """Create a video from a sequence of plots."""

    @abstractmethod
    def plot_frame(self, fig: plt.Figure, data: np.ndarray) -> plt.Figure:
        """Plot a single frame of data on a figure, and return the new figure.

        Subclasses should implement this method."""
        pass

    def plot_video(self, data_sequence: Iterable[np.ndarray], save_path: Union[Path, str], verbose: int = 0):
        """Plot video one frame at a time, using self.plot_frame"""
        with imageio.get_writer(str(save_path), mode="I", fps=30) as writer:
            for index, data in enumerate(data_sequence):
                if verbose >= 1 and index % 100 == 99:
                    print(f"Plotting frame {index+1}")
                fig = plt.figure()
                fig = self.plot_frame(fig, data)

                # Convert figure to an image
                fig.canvas.draw()
                io_buf = io.BytesIO()
                fig.savefig(io_buf, format="raw")
                io_buf.seek(0)
                img_arr = np.reshape(
                    np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                    newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
                )
                io_buf.close()

                writer.append_data(img_arr)
                plt.close(fig)
        if verbose >= 1:
            print("Finished plotting")
