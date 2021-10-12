from unittest import TestCase

import gym
import numpy as np
from blind_walking.envs.gym_envs import A1GymEnv


def test_local_terrain_view_sensor():
    env = A1GymEnv()
    grid_size = 32
    local_terrain_view = env.robot.GetLocalTerrainView(grid_unit=0.1, grid_size=grid_size)
    assert local_terrain_view.shape == (grid_size, grid_size)

    import matplotlib.pyplot as plt

    cs = [i for i in range(grid_size)]
    coords = [(x, y) for x in cs for y in cs]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    plt.figure()
    plt.scatter(xs, ys, c=local_terrain_view)
    plt.savefig("plot.png")
