import math

import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn


class LocomotionVisualEncoder(BaseFeaturesExtractor):
    """
    Combined visual encoder for augmented A1GymEnv-v0 observation space
    Assumes the observation space has two entries:
    -- 'flatten': gym.spaces.Box
    -- 'visual': gym.spaces.Box
    To be used with ObservationDictionarySplitByEncoderWrapper
    """

    def __init__(self, observation_space: gym.spaces.Dict, visual_output_size: int = 4):
        self.visual_output_size = visual_output_size
        flatten_output_size = math.prod(observation_space.spaces["flatten"].shape)
        features_dim = flatten_output_size + visual_output_size
        super().__init__(observation_space, features_dim=features_dim)

        self.flatten_encoder = nn.Flatten()
        cnn_layers = [
            nn.Conv2d(1, 32, 5, stride=1),
            nn.Conv2d(32, 32, 5, stride=4),
            nn.Flatten(),
            nn.Linear(128, visual_output_size),
        ]
        self.visual_encoder = nn.Sequential(*cnn_layers)

    def forward(self, observations: TensorDict) -> th.Tensor:
        assert set(observations.keys()) == set(["flatten", "visual"])
        flatten_embedding = self.flatten_encoder(observations["flatten"])
        visual_input = observations["visual"]
        visual_embedding = self.visual_encoder(visual_input)
        return th.cat([flatten_embedding, visual_embedding], dim=1)
