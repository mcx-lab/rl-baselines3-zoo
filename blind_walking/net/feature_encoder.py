import math

import gym
import torch
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn


class LocomotionFeatureEncoder(BaseFeaturesExtractor):
    """
    Combined feature extractor for augmented A1GymEnv-v0 observation space
    Assumes the observation space has two entries:
    -- 'flatten': gym.spaces.Box
    -- 'mlp': gym.spaces.Box

    To be used with ObservationDictionarySplitByEncoderWrapper
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        mlp_output_size: int = 8,
        cnn_output_size: int = 8,
    ):
        self.mlp_output_size = mlp_output_size
        flatten_output_size = math.prod(observation_space.spaces["flatten"].shape)
        assert len(observation_space.spaces["mlp"].shape) == 1, "Mlp-encoded space must be 1-dimensional"
        mlp_input_size = observation_space.spaces["mlp"].shape[0]
        features_dim = flatten_output_size + mlp_output_size + cnn_output_size

        super(LocomotionFeatureEncoder, self).__init__(observation_space, features_dim=features_dim)

        self.flatten_encoder = nn.Flatten()
        mlp_layers = create_mlp(input_dim=mlp_input_size, output_dim=mlp_output_size, net_arch=[128])
        self.mlp_encoder = nn.Sequential(*mlp_layers)

        cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 16, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        # Get the output shape by doing a forward pass
        _inp = torch.zeros(observation_space["cnn"]["LocalTerrainView_cnn"].shape).unsqueeze(0)
        _oup = cnn_backbone(_inp)
        _oup_dim = _oup.view(-1).shape[0]
        cnn_head = nn.Sequential(nn.Flatten(), nn.Linear(_oup_dim, cnn_output_size))
        self.cnn_encoder = nn.Sequential(cnn_backbone, cnn_head)

    def forward(self, observations: TensorDict) -> th.Tensor:
        flatten_embedding = self.flatten_encoder(observations["flatten"])
        mlp_embedding = self.mlp_encoder(observations["mlp"])
        cnn_embedding = self.cnn_encoder(observations["cnn"]["LocalTerrainView_cnn"])

        return th.cat([flatten_embedding, mlp_embedding, cnn_embedding], dim=1)
