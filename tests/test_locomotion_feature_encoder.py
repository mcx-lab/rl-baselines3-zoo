import unittest

import gym
import numpy as np
import torch
from blind_walking.envs.env_wrappers import observation_dictionary_split_by_encoder_wrapper as obs_split_wrapper
from blind_walking.envs.gym_envs.a1_gym_env import A1GymEnv
from blind_walking.envs.sensors import environment_sensors, robot_sensors
from blind_walking.net.feature_encoder import LocomotionFeatureEncoder


class TestLocomotionFeatureEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.env = A1GymEnv(
            robot_sensor_list=[robot_sensors.BaseVelocitySensor(convert_to_local_frame=True, exclude_z=True)],
            env_sensor_list=[environment_sensors.TargetPositionSensor(enc_name="mlp")],
            obs_wrapper=obs_split_wrapper.ObservationDictionarySplitByEncoderWrapper,
        )
        self.extractor = LocomotionFeatureEncoder(self.env.observation_space)

    def test_forward(self):
        obs = self.env.reset()
        # Observation is a 1-level dictionary of np arrays
        # Cast to tensor, dtype float32, and add batch dimension
        obs_tensor = {k: torch.from_numpy(v).to(torch.float32).view(1, -1) for k, v in obs.items()}
        features = self.extractor(obs_tensor)
