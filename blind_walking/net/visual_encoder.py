import math
from pathlib import Path
from typing import Optional

import gym
import torch as th
from blind_walking.net.utilities import conv_output_shape
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn

from utils import ALGOS


def load_encoder(model_path: Path) -> th.nn.Module:
    model = ALGOS["ppo"].load(model_path, env=None)
    return model.policy.features_extractor


class LocomotionVisualEncoder(BaseFeaturesExtractor):
    """
    Combined visual encoder for augmented A1GymEnv-v0 observation space
    Assumes the observation space has two entries:
    -- 'flatten': gym.spaces.Box
    -- 'visual': gym.spaces.Box
    To be used with ObservationDictionarySplitByEncoderWrapper
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        visual_output_size: int = 16,
        load_path: Optional[str] = None,
        freeze: bool = False,
    ):
        self.visual_output_size = visual_output_size
        flatten_output_size = math.prod(observation_space.spaces["flatten"].shape)
        visual_input_size = math.prod(observation_space.spaces["visual"].shape)
        features_dim = flatten_output_size + visual_output_size
        super().__init__(observation_space, features_dim=features_dim)

        self.flatten_encoder = nn.Flatten()
        # 1-layer MLP extractor
        self.visual_encoder = nn.Sequential(nn.Linear(visual_input_size, visual_output_size), nn.ReLU())

        # Load the saved encoder from a previous run
        if load_path is not None:
            print("Loading pretrained encoder")
            from copy import deepcopy

            initial_weights = deepcopy(self.visual_encoder.state_dict())
            pretrained_encoder = load_encoder(Path(load_path) / "ppo" / "A1GymEnv-v0_4" / "A1GymEnv-v0.zip")
            self.visual_encoder.load_state_dict(pretrained_encoder.visual_encoder.state_dict())

            # Some basic sanity checks to ensure weights were loaded
            for key, value in self.visual_encoder.state_dict().items():
                print(f"Checking parameters: {key}")
                assert key in initial_weights
                assert th.all(initial_weights[key] != value)
            print("Loaded weights successfully")

            print(f"Visual encoder parameters will be {'frozen' if freeze else 'learnable'}")
            for param in self.visual_encoder.parameters():
                param.requires_grad = not freeze

    def forward(self, observations: TensorDict) -> th.Tensor:
        assert set(observations.keys()) == set(["flatten", "visual"])
        flatten_embedding = self.flatten_encoder(observations["flatten"])
        visual_input = observations["visual"]
        visual_embedding = self.visual_encoder(visual_input)
        return th.cat([flatten_embedding, visual_embedding], dim=1)
