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

"""An env wrapper that splits and flattens the A1GymEnv observations to an array based on its encoder """
import collections

import gym
from blind_walking.envs.utilities import env_utils
from gym import spaces


def get_encoder_from_sensor_name(sensor_name: str):
    sensor_id, encoder_name = sensor_name.split("_")
    return encoder_name


def get_all_encoders_from_observation_space(observation_space: spaces.dict.Dict):
    encoders = set()
    for sensor_name, _sensor in observation_space.spaces.items():
        encoder_name = get_encoder_from_sensor_name(sensor_name)
        encoders.add(encoder_name)
    return encoders


def filter_observation_space_by_encoder(observation_space: spaces.dict.Dict, encoder_name: str):
    filtered_observation_space = {}
    for sensor_name, sensor in observation_space.spaces.items():
        if encoder_name == get_encoder_from_sensor_name(sensor_name):
            filtered_observation_space[sensor_name] = sensor
    return spaces.dict.Dict(filtered_observation_space)


def get_all_encoders_from_observation(observation: dict):
    encoders = set()
    for sensor_name, _sensor in observation.items():
        encoder_name = get_encoder_from_sensor_name(sensor_name)
        encoders.add(encoder_name)
    return encoders


def filter_observation_by_encoder(observation: dict, encoder_name: str):
    filtered_observation = collections.OrderedDict()
    for sensor_name, sensor in observation.items():
        if encoder_name == get_encoder_from_sensor_name(sensor_name):
            filtered_observation[sensor_name] = sensor
    return filtered_observation


class ObservationDictionarySplitByEncoderWrapper(gym.Env):
    """An env wrapper that splits and flattens the observation dictionary to an array based on its encoder.

    Args:
        observation_excluded: A list of observations to skip splitting and flattening.
        These will be passed directly to the final output.
    """

    def __init__(self, gym_env, encoder_excluded=(), observation_excluded=()):
        """Initializes the wrapper."""
        self.encoder_excluded = encoder_excluded
        self.observation_excluded = observation_excluded
        self._gym_env = gym_env
        self.observation_space = self._split_observation_spaces(self._gym_env.observation_space)
        self.action_space = self._gym_env.action_space

    def __getattr__(self, attr):
        return getattr(self._gym_env, attr)

    def _split_observation_spaces(self, observation_spaces):
        included_observation_spaces = spaces.dict.Dict(
            {k: v for k, v in observation_spaces.spaces.items() if k not in self.observation_excluded}
        )
        excluded_observation_spaces = spaces.dict.Dict(
            {k: v for k, v in observation_spaces.spaces.items() if k in self.observation_excluded}
        )

        encoder_names = get_all_encoders_from_observation_space(included_observation_spaces)
        split_space = {}
        for enc_name in encoder_names:
            subspace = filter_observation_space_by_encoder(observation_spaces, enc_name)
            if enc_name not in self.encoder_excluded:
                flattened_subspace = env_utils.flatten_observation_spaces(subspace)
                split_space[enc_name] = flattened_subspace
            else:
                split_space[enc_name] = subspace

        for name, space in excluded_observation_spaces.spaces.items():
            split_space[name] = space

        return spaces.dict.Dict(split_space)

    def _split_observation(self, input_observation):
        included_observations = {k: v for k, v in input_observation.items() if k not in self.observation_excluded}
        excluded_observations = {k: v for k, v in input_observation.items() if k in self.observation_excluded}

        encoder_names = get_all_encoders_from_observation(included_observations)
        split_obs = collections.OrderedDict()
        for enc_name in encoder_names:
            subobs = filter_observation_by_encoder(input_observation, enc_name)
            if enc_name not in self.encoder_excluded:
                flattened_subspace = env_utils.flatten_observations(subobs)
                split_obs[enc_name] = flattened_subspace
            else:
                split_obs[enc_name] = subobs

        for name, obs in excluded_observations.items():
            split_obs[name] = obs

        return split_obs

    def reset(self, initial_motor_angles=None, reset_duration=0.0):
        observation = self._gym_env.reset(initial_motor_angles=initial_motor_angles, reset_duration=reset_duration)
        return self._split_observation(observation)

    def step(self, action):
        """Steps the wrapped environment.
        Args:
          action: Numpy array. The input action from an NN agent.
        Returns:
          The tuple containing the flattened observation, the reward, the epsiode
            end indicator.
        """
        observation_dict, reward, done, _ = self._gym_env.step(action)
        return self._split_observation(observation_dict), reward, done, _

    def render(self, mode="human"):
        return self._gym_env.render(mode)
