import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, datapath: str = None, name: str = "plot"):
        self.data = np.load(datapath)
        self.name = name

    def plot(self, savedir: str = None):
        plt.figure()
        t = np.arange(self.data.shape[0])
        for i in range(self.data.shape[1]):
            plt.plot(t, self.data[:, i])
            plt.savefig(os.path.join(savedir, self.name))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-folder", help="input path to folder which holds encoder data", type=str, default="./")
args = parser.parse_args()

data_name = "true_extrinsics"
true_extrinsics_plotter = Plotter(os.path.join(args.input_folder, f"{data_name}.npy"), data_name)
true_extrinsics_plotter.plot(args.input_folder)
