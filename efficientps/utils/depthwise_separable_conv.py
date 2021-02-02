import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    DepthwiseSeparableConv from MobileNet, code largely inspire from mmcv
    DepthwiseSeparableConvModule but simplify
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 **kwargs):

        super(DepthwiseSeparableConv, self).__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            **kwargs)

        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
