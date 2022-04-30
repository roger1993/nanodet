import pytest
import torch
import torch.nn as nn

from nanodet.model.module.conv import DepthwiseConvModule


def test_depthwise_conv():
    norm_cfg = dict(type="BN")
    with pytest.raises(AssertionError):
        activation = dict(type="softmax")
        DepthwiseConvModule(4, 5, 2, activation=activation, norm_cfg=norm_cfg)

    with pytest.raises(AssertionError):
        DepthwiseConvModule(3, 5, 2, order=("norm", "conv", "act"), norm_cfg=norm_cfg)

    # test default config
    conv = DepthwiseConvModule(3, 5, 2, norm_cfg=norm_cfg)
    assert conv.with_norm
    assert conv.depthwise.groups == 3
    assert conv.pointwise.kernel_size == (1, 1)
    assert conv.act.__class__.__name__ == "ReLU"
    x = torch.rand(1, 3, 16, 16)
    output = conv(x)
    assert output.shape == (1, 5, 15, 15)

    # test norm_cfg
    conv = DepthwiseConvModule(3, 5, 2, norm_cfg=dict(type="BN"))
    assert isinstance(conv.dwnorm, nn.BatchNorm2d)
    assert isinstance(conv.pwnorm, nn.BatchNorm2d)

    x = torch.rand(1, 3, 16, 16)
    output = conv(x)
    assert output.shape == (1, 5, 15, 15)

    # test act_cfg
    conv = DepthwiseConvModule(
        3, 5, 3, padding=1, activation="LeakyReLU", norm_cfg=norm_cfg
    )
    assert conv.act.__class__.__name__ == "LeakyReLU"
    output = conv(x)
    assert output.shape == (1, 5, 16, 16)
