import argparse
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        # Plot heightmap sensor data for each foot on the same plot
        plotter = Plotter(f, basename + f"_foothm")
        plotter.plot(columns=47 - np.arange(4), ylim=(-3, 3), savedir=args.input_folder)
        # Plot heightmap sensor data for each foot on separate plots
        for i in range(4):
            plotter = Plotter(f, basename + f"_foothm{i}")
            plotter.plot(columns=[47 - i], ylim=(-3, 3), savedir=args.input_folder)
        print("Generated foot heightmap images")

        grid_size = (20, 1)
        grid_unit = 0.05
        num_timesteps = len(plotter.data)
        # Generate GIF of heightmap over time
        dirpath = args.input_folder
        if grid_size[0] == 1 or grid_size[1] == 1:
            # bar graph plot
            for i in range(num_timesteps):
                plt.figure()
                data = plotter.data[i][48 - 24 : 48 - 4]
                plt.bar(x=np.arange(len(data)), height=data)
                plt.ylim((-3, 3))
                plt.savefig(os.path.join(dirpath, f"tmp{i}"))
                plt.close()
            print("Generated images for video")
            # build gif
            files = glob.glob(os.path.join(dirpath, "tmp*.png"))
            with imageio.get_writer(os.path.join(dirpath, basename + "_hm.mp4"), mode="I", fps=30) as writer:
                for f in files:
                    image = imageio.imread(f)
                    writer.append_data(image)
            print("Created heightmap video")
            # remove images
            for f in files:
                os.remove(f)
            print("Removed unnessary image files")
        else:
            # scatter plot
            kx = grid_size[0] / 2 - 0.5
            xvalues = np.linspace(-kx * grid_unit, kx * grid_unit, num=grid_size[0])
            ky = grid_size[1] / 2 - 0.5
            yvalues = np.linspace(-ky * grid_unit, ky * grid_unit, num=grid_size[1])
            xx, yy = np.meshgrid(xvalues, yvalues)
            # generate images
            images = []
            dpi = 60
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            for i in range(num_timesteps):
                img = ax.scatter(xx, yy, c=plotter.data[i][48 - 24 : 48 - 4], vmin=-3, vmax=3)
                fig.colorbar(img, cax=cax, orientation="vertical")
                image = get_img_from_fig(fig, dpi=dpi)
                images.append(image)
            plt.close(fig)
            print("Generated images for video")
            # build gif
            heightmap_video_path = os.path.join(dirpath, basename + "_hm.mp4")
            with imageio.get_writer(heightmap_video_path, mode="I", fps=30) as writer:
                for image in images:
                    writer.append_data(image)
            print("Created heightmap video")

        if args.stitch_path:
            replay_video_path = args.stitch_path
            print(f"Stitching video from {replay_video_path}")
            replay_frames = get_frames_from_video_path(replay_video_path)[:1000]
            heightmap_frames = get_frames_from_video_path(heightmap_video_path)[:1000]

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
