from deep_learning.src.rcan import common
import torch.nn as nn
import math

# Adapted from implementation found below
# https://github.com/thstkdgus35/EDSR-PyTorch/blob/master/src/model/rcan.py

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN3D)
class RCAN(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, n_frames, n_groups, n_blocks, n_feats, kernel_size,
                 conv=common.default_conv):
        super(RCAN, self).__init__()
        in_channels = n_input_channels
        self.n_resgroups = n_groups
        self.n_resblocks = n_blocks
        self.n_feats = n_feats
        # 64: 12181.17
        self.kernel_size = kernel_size
        reduction = 1
        scale = 2
        act = nn.ReLU(True)

        # Round to nearest multiple of kernel size, subtract from input size
        conv3d_padding = (kernel_size - 1) // 2

        # define head module
        conv3d = nn.Conv3d(
            1, n_feats, (7 * n_frames, self.kernel_size, self.kernel_size),
            padding=(0, conv3d_padding, conv3d_padding))
        modules_head = [conv3d]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, self.n_feats, self.kernel_size, reduction, act=act, res_scale=scale, n_resblocks=self.n_resblocks) \
            for _ in range(self.n_resgroups)]

        modules_body.append(conv(self.n_feats, self.n_feats, self.kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, self.n_feats, act=False),
            conv(self.n_feats, n_output_channels, self.kernel_size),
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        x_shp = x.shape
        x = x.view(x_shp[0], x_shp[1] * x_shp[2], x_shp[3], x_shp[4])
        res = self.body(x)
        res += x

        x = self.tail(res)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
