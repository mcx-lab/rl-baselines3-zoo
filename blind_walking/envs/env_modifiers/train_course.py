import numpy as np
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier
from blind_walking.envs.env_modifiers.stairs import Stairs

""" Train robot to walk up stairs curriculum.

One easy set of stairs at the front, then one more difficult set of stairs.
Equal chances for the robot to encounter going up and going down the stairs.
"""


class TrainStairs(EnvModifier):
    def __init__(self):
        super().__init__()
        self.easy_stairs = Stairs()
        self.hard_stairs = Stairs()

    def _generate(self, env):
        self.easy_stairs._generate(env, start_x=1, step_rise=0.02)
        self.hard_stairs._generate(env, start_x=7, step_rise=0.05)

    def _reset(self, env):
        if np.random.uniform() < 0.5:
            self.adjust_position = (3, 0, 0.1)
        else:
            self.adjust_position = (0, 0, 0)
