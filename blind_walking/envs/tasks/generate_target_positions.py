import os
import math
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


# ######################### Parameters ######################### #

start_pos = [0, 0]
n_timesteps = 2000
filename_csv = "target_positions.csv"
filename_png = "target_positions.png"
filepath_csv = os.path.join(os.path.dirname(__file__), filename_csv)
filepath_png = os.path.join(os.path.dirname(__file__), filename_png)

speed_default = 0.02
speed_timestep_signals = [1900, 1600, 1300, 1000]  # have to be in descending order
target_speeds = [0.0, 0.014, 0.016, 0.018]

dir_default = 0
dir_timestep_signals = [800, 600, 400, 200]  # have to be in descending order
target_dirs = [0, -0.2, 0, 0.2, 0]

plot_graph = True


# ######################### Helper Functions ######################### #


def get_target_speed(timestep):
    for i, t in enumerate(speed_timestep_signals):
        if timestep > t:
            return target_speeds[i]
    return speed_default


def get_target_dir(timestep):
    for i, t in enumerate(dir_timestep_signals):
        if timestep > t:
            return target_dirs[i]
    return dir_default


def calculate_target_pos(current_pos, target_speed, target_dir):
    # Calculates the x, y target position
    dx_target = target_speed * math.cos(target_dir)
    dy_target = target_speed * math.sin(target_dir)
    x_target = current_pos[0] + dx_target
    y_target = current_pos[1] + dy_target
    return np.around([x_target, y_target], 8)


# ######################### Main Script ######################### #

current_pos = start_pos
f = open(filepath_csv, "w")
writer = csv.writer(f, delimiter=",")
writer.writerow(current_pos)
for t in range(n_timesteps):
    # calculate target position
    target_speed = get_target_speed(t)
    target_dir = get_target_dir(t)
    target_pos = calculate_target_pos(current_pos, target_speed, target_dir)
    # write target position to csv file
    writer.writerow(target_pos)
    current_pos = target_pos
f.close()

if plot_graph:
    # Plot target positions for visualisation
    df = pd.read_csv(filepath_csv)
    df["distance"] = [
        np.linalg.norm([x, y], 2)
        for (x, y) in list(zip(df.iloc[:, 0].diff(), df.iloc[:, 1].diff()))
    ]
    fig = df.plot(
        kind="scatter",
        x=0,
        y=1,
        c="distance",
        colormap="plasma",
        xlabel="X",
        ylabel="Y",
        title="Target Positions",
    ).get_figure()
    fig.savefig(filepath_png)
