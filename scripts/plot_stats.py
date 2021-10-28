import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import glob


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
    # Plot heightmap sensor data for each foot
    for i in range(4):
        plotter = Plotter(f, os.path.splitext(os.path.basename(f))[0] + f"_foothm{i}")
        plotter.plot(columns=[47 - i], ylim=(-1, 3), savedir=args.input_folder)
