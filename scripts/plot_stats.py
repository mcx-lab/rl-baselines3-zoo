import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-folder", help="input path to folder which holds the stats data", type=str, default="./")
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

        # Avoid circuluar import
        from blind_walking.examples.hover_robot import get_img_from_fig

        grid_size = (20, 1)
        grid_unit = 0.05
        num_timesteps = 1000
        # Generate GIF of heightmap over time
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
            img = ax.scatter(xx, yy, c=plotter.data[i][48-24:48-4], vmin=-3, vmax=3)
            fig.colorbar(img, cax=cax, orientation="vertical")
            image = get_img_from_fig(fig, dpi=dpi)
            images.append(image)
        plt.close(fig)

        print("Generated images for video")
        # build gif
        dirpath = args.input_folder
        heightmap_video_path = os.path.join(dirpath, basename + "_hm.mp4")
        with imageio.get_writer(heightmap_video_path, mode="I", fps=30) as writer:
            for image in images:
                writer.append_data(image)
        print("Created heightmap video")
