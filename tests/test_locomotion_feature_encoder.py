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
            env_sensor_list=[
                environment_sensors.TargetPositionSensor(enc_name="mlp"),
                environment_sensors.LocalTerrainViewSensor(enc_name="cnn"),
            ],
            obs_wrapper=lambda x: obs_split_wrapper.ObservationDictionarySplitByEncoderWrapper(
                x, observation_excluded="LocalTerrainView_cnn"
            ),
        )
        self.extractor = LocomotionFeatureEncoder(self.env.observation_space)

    def test_forward(self):
        obs = self.env.reset()

        def nested_dict_map(d, f):
            """
            d: (possibly nested) dictionary
            f: function

            Returns a dictionary of identical structure with f(v) in place of v for all leaf values in d
            """
            d_mapped = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    d_mapped[k] = nested_dict_map(v, f)
                else:
                    d_mapped[k] = f(v)
            return d_mapped

        obs_tensor_dict = nested_dict_map(
            obs,
            # Observation is a np array
            # Cast to tensor, dtype float32, and add batch dimension
            lambda x: torch.from_numpy(x).to(torch.float32).unsqueeze(0),
        )
        features = self.extractor(obs_tensor_dict)
