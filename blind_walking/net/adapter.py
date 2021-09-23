import math
import gym
import torch as th
from torch import nn
from typing import List
from collections import deque
from stable_baselines3.common.type_aliases import TensorDict


# TODO - move this to a utilities file
def build_mlp(input_size: int, arch: List[int]):
    layer_sizes = [input_size] + arch
    layers = []
    for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class Adapter(nn.Module):
    '''
    Adapter module to imitate feature encoder. 
    Uses only 'flatten' observation space to predict extrinsics.
    '''

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_size=8):
        super(Adapter, self).__init__()

        self.history_length = 50
        self.recent_states = deque([])
        obs_flatten_input_size = math.prod(observation_space.spaces['flatten'].shape)

        self.flatten_encoder = nn.Flatten()
        self.adapter_mlp_encoder = build_mlp(obs_flatten_input_size, [256, 32])
        self.adapter_cnn_encoder = [nn.Conv1d(32, 32, 5, stride=1),
                                    nn.Conv1d(32, 32, 5, stride=1),
                                    nn.Conv1d(32, 32, 8, stride=4),
                                    nn.Flatten(),
                                    nn.Linear(288, cnn_output_size)]

    def forward(self, observations: TensorDict):
        # Update states and actions stored
        self.recent_states.append(observations['flatten'])
        while len(self.recent_states) < self.history_length:
            self.recent_states.append(observations['flatten'])
        if len(self.recent_states) > self.history_length:
            self.recent_states.popleft()

        # Embed recent states and actions
        recent_states_embedding = []
        for s in self.recent_states:
            t = self.adapter_mlp_encoder(s)
            recent_states_embedding.append(t)

        # Predict extrinsics using adapter
        layer_input = th.stack(recent_states_embedding, dim=2)
        for layer in self.adapter_cnn_encoder:
            layer_output = layer(layer_input)
            layer_input = layer_output
        extrinsics = layer_output

        # Add flatten embedding
        flatten_embedding = self.flatten_encoder(observations['flatten'])
        extrinsics_and_observations = th.cat([flatten_embedding, extrinsics], dim=1)

        return extrinsics_and_observations
