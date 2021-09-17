import gym
import math

import torch as th
from torch import nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from blind_walking.envs.utilities.env_utils import nested_dict_space_iter

class A1GymEnvCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for augmented A1GymEnv-v0 observation space

    Assumes the observation space has two entries: 
    -- 'robot_observations': gym.spaces.Box
    -- 'environment_state': gym.spaces.Box
    """

    def __init__(self, 
        observation_space: gym.spaces.Dict,
        env_state_encoder_hidden_dim: int = 256, 
        env_state_encoder_output_dim: int = 16
    ):
        assert set(observation_space.spaces.keys()) == set(['robot_observations', 'environment_state'])  
        
        robot_observations_size = math.prod(observation_space.spaces['robot_observations'].shape)
        environment_state_size = math.prod(observation_space.spaces['environment_state'].shape)
        features_dim = robot_observations_size + env_state_encoder_output_dim  
        super(A1GymEnvCombinedExtractor, self).__init__(observation_space, features_dim=features_dim)

        self.robot_observations_encoder = lambda x: x
        self.environment_state_encoder = nn.Sequential(
            nn.Linear(in_features=environment_state_size, out_features=env_state_encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=env_state_encoder_hidden_dim, out_features=env_state_encoder_output_dim)
        )

    def forward(self, observations: TensorDict) -> th.Tensor:
        assert set(observations.keys()) == set(['robot_observations', 'environment_state'])

        robot_observations_embedding = self.robot_observations_encoder(observations['robot_observations'])
        environment_state_embedding = self.environment_state_encoder(observations['environment_state'])

        return th.cat([
            robot_observations_embedding,
            environment_state_embedding
        ], dim=1)