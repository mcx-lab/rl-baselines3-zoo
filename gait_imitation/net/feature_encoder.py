import math
import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn

from train_autoencoder import LinearAE


class LocomotionFeatureEncoder(BaseFeaturesExtractor):
    """
    Combined feature extractor for augmented A1GymEnv-v0 observation space
    Assumes the observation space has two entries:
    -- 'flatten': gym.spaces.Box
    -- 'enc': gym.spaces.Box

    To be used with ObservationDictionarySplitByEncoderWrapper
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        enc_output_size: int = 32,
        load_path: str = None,
    ):
        self.enc_output_size = enc_output_size
        flatten_output_size = math.prod(observation_space.spaces["flatten"].shape)
        assert len(observation_space.spaces["enc"].shape) == 1, "Specified encoded space must be 1-dimensional"
        enc_input_size = observation_space.spaces["enc"].shape[0]
        features_dim = flatten_output_size + enc_output_size
        super(LocomotionFeatureEncoder, self).__init__(observation_space, features_dim=features_dim)

        # flatten net
        self.flatten_net = nn.Flatten()
        # encoder net
        if load_path:
            # load pretrained encoder from path
            model = LinearAE(input_size=enc_input_size, code_size=enc_output_size)
            model_state, optimizer_state = th.load(load_path)
            model.load_state_dict(model_state)
            model.eval()
            self.encoder_net = model.encoder
            for param in self.encoder_net.parameters():
                param.requires_grad = False
        else:
            # create mlp encoder to be trained
            enc_layers = create_mlp(input_dim=enc_input_size, output_dim=enc_output_size, net_arch=[256])
            self.encoder_net = nn.Sequential(*enc_layers)

    def forward(self, observations: TensorDict) -> th.Tensor:
        assert set(observations.keys()) == set(["flatten", "enc"])

        flatten_embedding = self.flatten_net(observations["flatten"])
        enc_embedding = self.encoder_net(observations["enc"])

        return th.cat([flatten_embedding, enc_embedding], dim=1)
