import gym
import math

import torch as th
from typing import List
from torch import nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor


def build_mlp(input_size: int, arch: List[int]):
    layer_sizes = [input_size] + arch
    layers = []
    for index, (input_size, output_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(input_size, output_size))
        is_last_layer = index == len(layer_sizes) - 1
        if not is_last_layer:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)    

class A1GymEnvFeaturesExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for augmented A1GymEnv-v0 observation space
    Assumes the observation space has two entries: 
    -- 'flatten': gym.spaces.Box
    -- 'mlp': gym.spaces.Box
    """

    def __init__(self, 
                 observation_space: gym.spaces.Dict,
                 mlp_arch = [256, 16]):
        flatten_output_size = math.prod(observation_space.spaces['flatten'].shape)
        assert len(observation_space.spaces['mlp'].shape) == 1, "Mlp-encoded space must be 1-dimensional"
        mlp_input_size = observation_space.spaces['mlp'].shape[0]
        features_dim = flatten_output_size + mlp_arch[-1]  
        super(A1GymEnvFeaturesExtractor, self).__init__(observation_space, features_dim=features_dim)

        self.flatten_encoder = nn.Flatten() if len(observation_space.spaces['flatten'].shape) > 1 else lambda x: x
        self.mlp_encoder = build_mlp(mlp_input_size, mlp_arch)
        
    def forward(self, observations: TensorDict) -> th.Tensor:
        assert set(observations.keys()) == set(['flatten', 'mlp'])

        flatten_embedding = self.flatten_encoder(observations['flatten'])
        mlp_embedding = self.mlp_encoder(observations['mlp'])

        return th.cat([
            flatten_embedding,
            mlp_embedding
        ], dim=1)
