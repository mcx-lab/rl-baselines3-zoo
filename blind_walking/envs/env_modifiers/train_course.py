import numpy as np
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier
from blind_walking.envs.env_modifiers.stairs import Stairs, boxHalfLength, boxHalfWidth
from blind_walking.envs.env_modifiers.heightfield import HeightField

""" Train robot to walk up stairs curriculum.

Equal chances for the robot to encounter going up and going down the stairs.
"""


class TrainStairs(EnvModifier):
    def __init__(self):
        super().__init__()
        self.step_rise_levels = [0.02, 0.05, 0.075, 0.1, 0.125, 0.15]
        self.num_levels = len(self.step_rise_levels)
        self.num_steps = 5
        self.stair_gap = 1.5
        self.step_run = 0.3
        self.stair_length = (self.num_steps - 1) * self.step_run * 2 + boxHalfLength * 2 * 2

        self._level = 0

        self.stairs = []
        for _ in range(self.num_levels):
            self.stairs.append(Stairs())

    def _generate(self, env):
        start_x = self.stair_gap
        for i in range(self.num_levels):
            self.stairs[i]._generate(
                env, start_x=start_x, num_steps=self.num_steps, step_rise=self.step_rise_levels[i], step_run=self.step_run
            )
            start_x += self.stair_length + self.stair_gap

    def _reset(self, env):
        # Check if robot has succeeded current level
        if self._level < self.num_levels and self.succeed_level(env):
            print(f"LEVEL {self._level} PASSED!")
            self._level += 1
        level = self._level
        if level >= self.num_levels:
            # Loop back to randomly selected level
            level_list = np.arange(self.num_levels)
            level_probs = level_list / sum(level_list)
            level = np.random.choice(self.num_levels, p=level_probs)
            print(f"LOOP TO LEVEL {level}")
        elif level > 0 and np.random.uniform() < 0.2:
            # Redo previous level
            level -= 1

        x_pos = level * (self.stair_length + self.stair_gap)
        z_pos = 0
        # Equal chances to encouter going up and down the stair level
        if np.random.uniform() < 0.4:
            x_pos += self.stair_gap + self.stair_length / 2 - 1
            z_pos = self.step_rise_levels[level] * self.num_steps
        self.adjust_position = (x_pos, 0, z_pos)

    def succeed_level(self, env):
        """To succeed the current level, robot needs to climb over the current stair level and reach the start of next stair level"""
        base_pos = env._robot.GetBasePosition()
        target_x = (self._level + 1) * (self.stair_length + self.stair_gap) + 0.5
        return (
            self.adjust_position[2] == 0
            and base_pos[0] > target_x
            and base_pos[1] > -boxHalfWidth
            and base_pos[1] < boxHalfWidth
        )


class TrainUneven(EnvModifier):
    def __init__(self):
        super().__init__()
        self.hf = HeightField()

    def _generate(self, env):
        self.hf._generate(env, start_x=10, heightPerturbationRange=0.04)
