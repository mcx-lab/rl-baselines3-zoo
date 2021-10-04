import collections
import unittest

import gym
import numpy as np
from blind_walking.envs.env_wrappers import observation_dictionary_split_by_encoder_wrapper as obs_split_wrapper
from blind_walking.envs.gym_envs.a1_gym_env import A1GymEnv
from blind_walking.envs.sensors import environment_sensors, robot_sensors


class TestObservationDictionarySplitByEncoderWrapper(unittest.TestCase):
    def test_order(self):
        """Test that the observation dictionary space has consistent ordering"""
        velocity_sensor = robot_sensors.BaseVelocitySensor(convert_to_local_frame=True, exclude_z=True)
        target_pos_sensor = environment_sensors.TargetPositionSensor(enc_name="mlp")
        wrapped_env = A1GymEnv(
            robot_sensor_list=[velocity_sensor],
            env_sensor_list=[target_pos_sensor],
            obs_wrapper=obs_split_wrapper.ObservationDictionarySplitByEncoderWrapper,
        )

        print(velocity_sensor._lower_bound, velocity_sensor._upper_bound)
        for i in range(1000):
            obs = wrapped_env.reset()
            print(obs)
            assert np.all(velocity_sensor._upper_bound >= obs["flatten"])
            assert np.all(obs["flatten"] >= velocity_sensor._lower_bound)
            assert np.all(target_pos_sensor._upper_bound >= obs["mlp"])
            assert np.all(obs["mlp"] >= target_pos_sensor._lower_bound)
