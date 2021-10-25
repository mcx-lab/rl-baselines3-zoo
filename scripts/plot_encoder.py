import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import glob


class Plotter:
    def __init__(self, datapath: str = None, name: str = "plot"):
        self.data = np.load(datapath)
        self.name = name

    def plot(self, savedir: str = None):
        plt.figure()
        t = np.arange(self.data.shape[0])
        for i in range(self.data.shape[1]):
            plt.plot(t, self.data[:, i])
            plt.ylim(-10, 10)
            plt.savefig(os.path.join(savedir, self.name))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-folder", help="input path to folder which holds encoder data", type=str, default="./")
args = parser.parse_args()

data_name = "true_extrinsics"
files = glob.glob(os.path.join(args.input_folder, f"{data_name}*.npy"))
for f in files:
    true_extrinsics_plotter = Plotter(f, os.path.splitext(os.path.basename(f))[0])
    true_extrinsics_plotter.plot(args.input_folder)
