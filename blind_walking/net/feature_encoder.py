import gym
import math
import torch as th
from torch import nn
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp

def _create_depthwise_separable_conv2d_block(in_channels, out_channels):
    """ Create a depthwise-separable convolutional block used in MobileNet
    
    Reference: Figure 3 of https://arxiv.org/pdf/1704.04861.pdf """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels),
        nn.BatchNorm2d(),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.BatchNorm2d(),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def create_cnn( in_channels: int, 
                out_channels: int, 
                net_arch):
    layers = []
    layer_sizes = [in_channels] + net_arch + [out_channels]
    for inp, oup in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(_create_depthwise_separable_conv2d_block(inp, oup))
    return layers

class LocomotionFeatureEncoder(BaseFeaturesExtractor):
    """
    Combined feature extractor for augmented A1GymEnv-v0 observation space
    Assumes the observation space has two entries: 
    -- 'flatten': gym.spaces.Box
    -- 'mlp': gym.spaces.Box

    To be used with ObservationDictionarySplitByEncoderWrapper
    """

    def __init__(self, 
        observation_space: gym.spaces.Dict, 
        mlp_output_size: int = 8, 
    ):
        self.mlp_output_size = mlp_output_size
        flatten_output_size = math.prod(observation_space.spaces['flatten'].shape)
        assert len(observation_space.spaces['mlp'].shape) == 1, "Mlp-encoded space must be 1-dimensional"
        mlp_input_size = observation_space.spaces['mlp'].shape[0]
        features_dim = flatten_output_size + mlp_output_size
        super(LocomotionFeatureEncoder, self).__init__(observation_space, features_dim=features_dim)

        self.flatten_encoder = nn.Flatten()
        mlp_layers = create_mlp(input_dim=mlp_input_size,
                                output_dim=mlp_output_size,
                                net_arch=[256, 128])
        self.mlp_encoder = nn.Sequential(*mlp_layers)

    def forward(self, observations: TensorDict) -> th.Tensor:
        assert set(observations.keys()) == set(['flatten', 'mlp'])

        flatten_embedding = self.flatten_encoder(observations['flatten'])
        mlp_embedding = self.mlp_encoder(observations['mlp'])

        return th.cat([flatten_embedding, mlp_embedding], dim=1)
