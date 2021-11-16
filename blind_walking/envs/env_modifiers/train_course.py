import numpy as np
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier
from blind_walking.envs.env_modifiers.heightfield import HeightField
from blind_walking.envs.env_modifiers.stairs import Stairs, boxHalfLength, boxHalfWidth

""" Train robot to walk up stairs curriculum.

Equal chances for the robot to encounter going up and going down the stairs.
"""


class TrainStairs(EnvModifier):
    def __init__(self):
        super().__init__()
        self.step_rise_levels = [0.02, 0.05]
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
            level_list = np.arange(self.num_levels) + 1
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
        """To succeed the current level, robot needs to climb over the current stair level
        and reach the start of next stair level"""
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
        self.hf._generate(env, start_x=10, heightPerturbationRange=0.08)


class TrainMultiple(EnvModifier):
    def __init__(self):
        super().__init__()

        self.hf_length = 20
        self.hf_perturb = 0.08
        self.hf = HeightField()

        self.step_rise_levels = [0.02, 0.05]
        self.num_levels = len(self.step_rise_levels)
        self.num_steps = 10
        self.stair_gap = 1.5
        self.step_run = 0.3
        self.stair_length = (self.num_steps - 1) * self.step_run * 2 + boxHalfLength * 2 * 2
        self._stair_level = 0
        self.stairs = []
        for _ in range(self.num_levels):
            self.stairs.append(Stairs())
        self._reset_manual_override = None

    def _generate(self, env):
        self.hf._generate(env, start_x=10, heightPerturbationRange=self.hf_perturb)
        start_x = self.stair_gap + self.hf_length
        for i in range(self.num_levels):
            self.stairs[i]._generate(
                env, start_x=start_x, num_steps=self.num_steps, step_rise=self.step_rise_levels[i], step_run=self.step_run
            )
            start_x += self.stair_length + self.stair_gap

    def _reset_to_heightfield(self):
        """Reset position to before the heightfield"""
        self.adjust_position = (0, 0, 0)

    def _select_stairs_level(self, env):
        if self._stair_level < self.num_levels and self.succeed_level(env):
            print(f"LEVEL {self._stair_level} PASSED!")
            self._stair_level += 1
        level = self._stair_level
        if level >= self.num_levels:
            # Loop back to randomly selected level
            level_list = np.arange(self.num_levels) + 1
            level_probs = level_list / sum(level_list)
            level = np.random.choice(self.num_levels, p=level_probs)
            print(f"LOOP TO LEVEL {level}")
        elif level > 0 and np.random.uniform() < 0.2:
            # Redo previous level
            level -= 1
        return level

    def _reset_to_stairs(self, level):
        """Reset position to just before the stairs of a given level"""
        x_pos = self.hf_length + level * (self.stair_length + self.stair_gap)
        z_pos = 0
        # Equal chances to encouter going up and down the stair level
        if np.random.uniform() < 0.4:
            x_pos += self.stair_gap + self.stair_length / 2 - 1
            z_pos = self.step_rise_levels[level] * self.num_steps
        self.adjust_position = (x_pos, 0, z_pos)

    def _reset_randomly(self, env):
        if np.random.uniform() < 0.5:
            # See heightfield
            self._reset_to_heightfield()
        else:
            # See stairs
            # Check if robot has succeeded current level
            level = self._select_stairs_level(env)
            self._reset_to_stairs(level)

    def _reset(self, env):
        if self._reset_manual_override is not None:
            self._reset_manually()
            # Remove override for subsequent resets
            # self._reset_manual_override = None
        else:
            self._reset_randomly(env)

    def _reset_manually(self):
        if self._reset_manual_override == "heightfield":
            self._reset_to_heightfield()
        elif self._reset_manual_override == "stairs_0":
            self._reset_to_stairs(level=0)
        elif self._reset_manual_override == "stairs_1":
            self._reset_to_stairs(level=1)
        else:
            raise ValueError(f"Invalid override {self._reset_manual_override}")

    def _override_reset(self, override: str):
        """Manually set what the next reset should be"""
        assert override in ("heightfield", "stairs_0", "stairs_1")
        self._reset_manual_override = override

    def succeed_level(self, env):
        """To succeed the current level, robot needs to climb over the current stair level
        and reach the start of next stair level"""
        base_pos = env._robot.GetBasePosition()
        target_x = self.hf_length + (self._stair_level + 1) * (self.stair_length + self.stair_gap) + 0.5
        return (
            self.adjust_position[2] == 0
            and base_pos[0] > target_x
            and base_pos[1] > -boxHalfWidth
            and base_pos[1] < boxHalfWidth
        )
