import gym 
import torch
import numpy as np
import unittest

from gym import spaces
from blind_walking.envs.gym_envs.a1_gym_env import A1GymEnv
from blind_walking.net.feature_extractor import A1GymEnvCombinedExtractor

class TestA1GymEnv(unittest.TestCase):
    def test_observation_space(self):
        env = A1GymEnv()
        obs_space = env.observation_space

        assert set(obs_space.spaces.keys()) == set(['robot_observations', 'environment_state'])
        assert isinstance(obs_space.spaces['robot_observations'], spaces.Box)
        assert isinstance(obs_space.spaces['environment_state'], spaces.Box)

    def test_default_env_parameters(self):
        env = A1GymEnv()
        assert np.all(env._env._gym_env._robot.GetMotorPositionGains() == 55.0)
        assert np.all(env._env._gym_env._robot.GetMotorVelocityGains() == 0.6)
        assert np.all(env._env._gym_env._robot.GetMotorStrengthRatios() == 1)

class TestA1GymEnvCombinedExtractor(unittest.TestCase):
    def test_forward(self):
        env = A1GymEnv()
        obs = env.reset()
        # Observation is a 1-level dictionary of np arrays
        # Cast to tensor, dtype float32, and add batch dimension
        obs_tensor = {k: torch.from_numpy(v).to(torch.float32).view(1,-1) for k, v in obs.items()}
        extractor = A1GymEnvCombinedExtractor(env.observation_space)
        features = extractor(obs_tensor)

