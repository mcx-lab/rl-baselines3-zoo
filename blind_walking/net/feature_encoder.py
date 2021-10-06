import math

import gym
import torch
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn


def contains_subspace(observation_space: gym.spaces.Dict, subspace_name: str):
    return subspace_name in observation_space.spaces.keys()


class LocomotionFeatureEncoder(BaseFeaturesExtractor):
    """
    Combined feature extractor for augmented A1GymEnv-v0 observation space

    Assumes the observation space always contains:
    -- 'flatten': gym.spaces.Box

    Observation space may optionally contain:
    -- 'mlp': gym.spaces.Box
    -- 'LocalTerrainView_cnn': gym.spaces.Box

    To be used with ObservationDictionarySplitByEncoderWrapper
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        mlp_output_size: int = 8,
        cnn_output_size: int = 8,
    ):
        features_dim = 0

        self.flatten_output_size = math.prod(observation_space.spaces["flatten"].shape)
        features_dim += self.flatten_output_size

        self._contains_mlp = contains_subspace(observation_space, "mlp")
        self._contains_cnn = contains_subspace(observation_space, "LocalTerrainView_cnn")

        if self._contains_mlp:
            assert len(observation_space.spaces["mlp"].shape) == 1, "Mlp-encoded space must be 1-dimensional"
            self.mlp_output_size = mlp_output_size
            mlp_input_size = observation_space.spaces["mlp"].shape[0]
            features_dim += self.mlp_output_size

        if self._contains_cnn:
            self.cnn_output_size = cnn_output_size
            features_dim += self.cnn_output_size

        super(LocomotionFeatureEncoder, self).__init__(observation_space, features_dim=features_dim)

        self.flatten_encoder = nn.Flatten()

        if self._contains_mlp:
            mlp_layers = create_mlp(input_dim=mlp_input_size, output_dim=mlp_output_size, net_arch=[128])
            self.mlp_encoder = nn.Sequential(*mlp_layers)

        if self._contains_cnn:
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
            _inp = torch.zeros(observation_space["LocalTerrainView_cnn"].shape).unsqueeze(0)
            _oup = cnn_backbone(_inp)
            _oup_dim = _oup.view(-1).shape[0]
            cnn_head = nn.Sequential(nn.Flatten(), nn.Linear(_oup_dim, cnn_output_size))
            self.cnn_encoder = nn.Sequential(cnn_backbone, cnn_head)

    def forward(self, observations: TensorDict) -> th.Tensor:
        embeddings = []

        flatten_embedding = self.flatten_encoder(observations["flatten"])
        embeddings.append(flatten_embedding)
        if self._contains_mlp:
            mlp_embedding = self.mlp_encoder(observations["mlp"])
            embeddings.append(mlp_embedding)
        if self._contains_cnn:
            cnn_embedding = self.cnn_encoder(observations["LocalTerrainView_cnn"])
            embeddings.append(cnn_embedding)

        return th.cat(embeddings, dim=1)
