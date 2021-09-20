import gym 
import torch
import numpy as np
import unittest

from gym import spaces
from blind_walking.envs.gym_envs.a1_gym_env import A1GymEnv
from blind_walking.net.a1_gym_env_features_extractor import A1GymEnvFeaturesExtractor

class TestA1GymEnv(unittest.TestCase):
    def test_default_env_parameters(self):
        env = A1GymEnv()
        assert np.all(env._env._gym_env._robot.GetMotorPositionGains() == 55.0)
        assert np.all(env._env._gym_env._robot.GetMotorVelocityGains() == 0.6)
        assert np.all(env._env._gym_env._robot.GetMotorStrengthRatios() == 1)

class TestA1GymEnvFeaturesExtractor(unittest.TestCase):
    def test_forward(self):
        env = A1GymEnv()
        obs = env.reset()
        # Observation is a 1-level dictionary of np arrays
        # Cast to tensor, dtype float32, and add batch dimension
        obs_tensor = {k: torch.from_numpy(v).to(torch.float32).view(1,-1) for k, v in obs.items()}
        extractor = A1GymEnvFeaturesExtractor(env.observation_space)
        features = extractor(obs_tensor)