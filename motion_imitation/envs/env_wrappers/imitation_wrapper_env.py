# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A wrapper for motion imitation environment."""

from __future__ import absolute_import, division, print_function

import gym
import numpy as np


class ImitationWrapperEnv(object):
    """An env using for training policy with motion imitation."""

    def __init__(self, gym_env):
        """Initialzes the wrapped env.

        Args:
          gym_env: An instance of LocomotionGymEnv.
        """
        self._gym_env = gym_env
        self.observation_space = self._build_observation_space()
        return

    def __getattr__(self, attr):
        return getattr(self._gym_env, attr)

    def step(self, action):
        """Steps the wrapped environment.

        Args:
          action: Numpy array. The input action from an NN agent.

        Returns:
          The tuple containing the modified observation, the reward, the epsiode end
          indicator.

        Raises:
          ValueError if input action is None.

        """
        original_observation, reward, done, info = self._gym_env.step(action)
        observation = self._modify_observation(original_observation)

        return observation, reward, done, info

    def reset(self, initial_motor_angles=None, reset_duration=0.0):
        """Resets the robot's position in the world or rebuild the sim world.

        The simulation world will be rebuilt if self._hard_reset is True.

        Args:
          initial_motor_angles: A list of Floats. The desired joint angles after
            reset. If None, the robot will use its built-in value.
          reset_duration: Float. The time (in seconds) needed to rotate all motors
            to the desired initial values.

        Returns:
          A numpy array contains the initial observation after reset.
        """
        original_observation = self._gym_env.reset(initial_motor_angles, reset_duration)
        observation = self._modify_observation(original_observation)

        return observation

    def _modify_observation(self, original_observation):
        """Appends target observations from the reference motion to the observations.

        Args:
          original_observation: A numpy array containing the original observations.

        Returns:
          A numpy array contains the initial original concatenated with target
          observations from the reference motion.
        """
        target_observation = self._task.build_target_obs()
        observation = np.concatenate([original_observation, target_observation], axis=-1)
        return observation

    def _build_observation_space(self):
        """Constructs the observation space, including target observations from
        the reference motion.

        Returns:
          Observation space representing the concatenations of the original
          observations and target observations.
        """
        obs_space0 = self._gym_env.observation_space
        low0 = obs_space0.low
        high0 = obs_space0.high

        task_low, task_high = self._task.get_target_obs_bounds()
        low = np.concatenate([low0, task_low], axis=-1)
        high = np.concatenate([high0, task_high], axis=-1)

        obs_space = gym.spaces.Box(low, high)

        return obs_space
