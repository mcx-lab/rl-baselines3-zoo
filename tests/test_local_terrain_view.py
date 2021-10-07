from unittest import TestCase

import gym
import numpy as np
from blind_walking.envs.gym_envs import A1GymEnv


def test_local_terrain_view_sensor():
    env = A1GymEnv()
    grid_size = (32, 32)
    grid_unit = 0.1
    local_terrain_view = env.robot.GetLocalTerrainView(grid_unit=grid_unit, grid_size=grid_size, transform=(0, 0))
    assert local_terrain_view.shape == grid_size

    import matplotlib.pyplot as plt

    kx = grid_size[0] / 2 - 0.5
    xvalues = np.linspace(-kx * grid_unit, kx * grid_unit, num=grid_size[0])
    ky = grid_size[1] / 2 - 0.5
    yvalues = np.linspace(-ky * grid_unit, ky * grid_unit, num=grid_size[1])
    xx, yy = np.meshgrid(xvalues, yvalues)
    plt.figure()
    plt.scatter(xx, yy, c=local_terrain_view)
    plt.savefig("plot.png")
