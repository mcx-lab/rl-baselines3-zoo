import math
from collections import deque

import gym
import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn


class Adapter(nn.Module):
    """
    Adapter module to imitate feature encoder.
    Uses only 'flatten' observation space to predict extrinsics.
    """

    def __init__(self, observation_space: gym.spaces.Dict, output_size=8):
        super(Adapter, self).__init__()

        self.history_length = 50
        self.recent_states = deque([])
        obs_flatten_input_size = math.prod(observation_space.spaces["flatten"].shape)

        self.flatten_encoder = nn.Flatten()
        adapter_mlp_layers = create_mlp(input_dim=obs_flatten_input_size, output_dim=32, net_arch=[256])
        self.adapter_mlp_encoder = nn.Sequential(*adapter_mlp_layers)
        adapter_cnn_layers = [
            nn.Conv1d(32, 32, 5, stride=1),
            nn.Conv1d(32, 32, 5, stride=1),
            nn.Conv1d(32, 32, 8, stride=4),
            nn.Flatten(),
            nn.Linear(288, output_size),
        ]
        self.adapter_cnn_encoder = nn.Sequential(*adapter_cnn_layers)

    def forward(self, observations: TensorDict):
        # Update states and actions stored
        self.recent_states.append(observations["flatten"])
        while len(self.recent_states) < self.history_length:
            self.recent_states.append(observations["flatten"])
        if len(self.recent_states) > self.history_length:
            self.recent_states.popleft()

        # Embed recent states and actions
        recent_states_embedding = []
        for s in self.recent_states:
            t = self.adapter_mlp_encoder(s)
            recent_states_embedding.append(t)

        # Predict extrinsics using adapter
        recent_states_embedding_tensor = th.stack(recent_states_embedding, dim=2)
        extrinsics = self.adapter_cnn_encoder(recent_states_embedding_tensor)

        # Add flatten embedding
        flatten_embedding = self.flatten_encoder(observations["flatten"])
        extrinsics_and_observations = th.cat([flatten_embedding, extrinsics], dim=1)

        return extrinsics_and_observations
