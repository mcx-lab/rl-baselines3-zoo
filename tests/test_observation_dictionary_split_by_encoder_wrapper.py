import numpy as np
import unittest
import collections
import gym

from blind_walking.envs.env_wrappers.observation_dictionary_split_by_encoder_wrapper import (
    ObservationDictionarySplitByEncoderWrapper,
)


class DummyA1GymEnv(gym.Env):
    """A stub for A1GymEnv in order to test ObservationDictionarySplitByEncoderWrapper"""

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(DummyA1GymEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(12,))
        self.observation_space = gym.spaces.Dict(
            {
                "01_flatten": gym.spaces.Box(low=0, high=1, shape=(2,)),
                "12_flatten": gym.spaces.Box(low=1, high=2, shape=(2,)),
                "23_mlp": gym.spaces.Box(low=2, high=3, shape=(2,)),
                "34_mlp": gym.spaces.Box(low=3, high=4, shape=(2,)),
            }
        )

    def step(self, action):
        return self.observation_space.sample()

    def reset(self, initial_motor_angles=None, reset_duration=0.0):
        return self.observation_space.sample()


class TestA1GymEnv(unittest.TestCase):
    def test_order(self):
        """Test that the observation dictionary space has consistent ordering"""
        env = DummyA1GymEnv()
        wrapped_env = ObservationDictionarySplitByEncoderWrapper(env)
        flatten_low = np.array([0, 0, 1, 1])
        flatten_high = np.array([1, 1, 2, 2])
        mlp_low = np.array([2, 2, 3, 3])
        mlp_high = np.array([3, 3, 4, 4])

        for i in range(1000):
            obs = wrapped_env.reset()
            assert np.all(flatten_high >= obs["flatten"])
            assert np.all(obs["flatten"] >= flatten_low)
            assert np.all(mlp_high >= obs["mlp"])
            assert np.all(obs["mlp"] >= mlp_low)
