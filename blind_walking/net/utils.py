from torch import nn
from typing import List


def build_mlp(input_size: int, arch: List[int]):
    layer_sizes = [input_size] + arch
    layers = []
    for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)  
