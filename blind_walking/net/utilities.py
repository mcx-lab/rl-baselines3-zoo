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
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def create_cnn(net_arch):
    layers = []
    layer_sizes = net_arch
    for inp, oup in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(_create_depthwise_separable_conv2d_block(inp, oup))
    return layers


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    if type(dilation) is not tuple:
        dilation = (dilation, dilation)
    h = (h_w[0] + (2 * pad[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w
