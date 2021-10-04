import unittest

import numpy as np
from blind_walking.robots.a1 import get_grid_coordinates


class TestGridCoordinates(unittest.TestCase):
    def test_get_grid_coordinates(self):
        coords = get_grid_coordinates(1.0, 3)
        coords = coords.reshape(3, 3, 2)
        assert np.all(coords[0, 0, :] == np.array([-1.0, -1.0]))
        assert np.all(coords[2, 2, :] == np.array([1.0, 1.0]))
        # x, y have the range (-1.0, 0, 1.0)
        # [0,2] corresponds to [-1.0, 1.0]
        assert np.all(coords[0, 2, :] == np.array([-1.0, 1.0]))
        assert np.all(coords[2, 0, :] == np.array([1.0, -1.0]))
