from torch import nn


def _create_depthwise_separable_conv2d_block(in_channels, out_channels):
    """Create a depthwise-separable convolutional block used in MobileNet

    Reference: Figure 3 of https://arxiv.org/pdf/1704.04861.pdf"""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            groups=in_channels,
        ),
        nn.BatchNorm2d(),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.BatchNorm2d(),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def create_cnn(in_channels: int, out_channels: int, net_arch):
    layers = []
    layer_sizes = [in_channels] + net_arch + [out_channels]
    for inp, oup in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(_create_depthwise_separable_conv2d_block(inp, oup))
    return layers
