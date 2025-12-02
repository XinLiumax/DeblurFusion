import torch
from torch import nn
from Networks import common
import torch.nn.functional as F


def head_conv_ir(in_channels=1, out_channels=32, kernel_size=1, bias=True, groups=1):
    conv_layer = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0, bias=bias, groups=groups
    )
    return conv_layer


def head_conv_vi(in_channels=1, out_channels=32, kernel_size=1, bias=True, groups=1):
    conv_layer = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0, bias=bias, groups=groups
    )
    return conv_layer


def convfuse(in_channels=34, out_channels=32, bias=True):
    conv_stack = nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(16, 8, kernel_size=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(4, 1, kernel_size=1, bias=bias),
        nn.Sigmoid(),
    )
    return conv_stack


def sr(out_channels, kernel_size=1, bias=True):
    conv_stack = nn.Sequential(
        nn.Conv2d(32, 16, kernel_size=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(16, 8, kernel_size=1, bias=bias),
        nn.Sigmoid(),
    )
    return conv_stack


def sr1(out_channels, kernel_size=1, bias=True):
    conv_stack = nn.Sequential(
        nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=bias),
        nn.Sigmoid(),
    )
    return conv_stack


def sr2(out_channels, kernel_size=1, bias=True):
    conv_stack = nn.Sequential(
        nn.Conv2d(8, 4, kernel_size=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(2, 1, kernel_size=1, bias=bias),
        nn.Sigmoid(),
    )
    return conv_stack


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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


class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3, relu=True):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size,
                              padding=kernel_size >> 1, stride=1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate, inchanels, kernel_size=kernel_size,
                               padding=kernel_size >> 1, stride=1)
        if relu:
            self.relu = nn.PReLU(growth_rate)

    def forward(self, x):
        output = self.conv1(self.conv(x))
        return output


class one_conv_1(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3, relu=True):
        super(one_conv_1, self).__init__()
        self.conv = nn.Conv2d(inchanels, inchanels // 2, kernel_size=kernel_size,
                              padding=kernel_size >> 1, stride=1)
        self.conv1 = nn.Conv2d(inchanels, inchanels // 2, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.conv1(x)
        output = torch.cat((out1, out2), 1)
        return output


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1,
                 dilation=1, groups=1, relu=True, bn=False, bias=False,
                 up_size=0, fan=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.in_channels = in_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class one_module(nn.Module):
    def __init__(self, n_feats):
        super(one_module, self).__init__()
        self.layer1 = one_conv_1(n_feats, n_feats // 2, 3)
        self.layer2 = one_conv_1(n_feats, n_feats // 2, 3)
        self.layer4 = BasicConv(n_feats, n_feats, 3, 1, 1)
        self.alise = BasicConv(2 * n_feats, n_feats, 1, 1, 0)
        self.atten = CALayer(n_feats)
        self.weight1 = common.Scale(1)
        self.weight2 = common.Scale(1)
        self.weight3 = common.Scale(1)
        self.weight4 = common.Scale(1)
        self.weight5 = common.Scale(1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2),
                                                          self.weight3(x1)], 1))))
        return self.weight4(x) + self.weight5(x4)


class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        self.encoder = one_module(n_feats)
        self.decoder_low = one_module(n_feats)
        self.decoder_high = one_module(n_feats)
        self.alise = one_module(n_feats)
        self.alise2 = BasicConv(2 * n_feats, n_feats, 1, 1, 0)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:],
                                  mode='bilinear', align_corners=True)

        for i in range(1):
            x2 = self.decoder_low(x2)

        high1 = self.decoder_high(high)
        x4 = F.interpolate(x2, size=x.size()[-2:],
                           mode='bilinear', align_corners=True)

        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x


class Un(nn.Module):
    def __init__(self, n_feats, wn):
        super(Un, self).__init__()
        self.encoder1 = Updownblock(n_feats)
        self.encoder2 = Updownblock(n_feats)
        self.reduce = common.default_conv(2 * n_feats, n_feats, 3)
        self.weight2 = common.Scale(1)
        self.weight1 = common.Scale(1)
        self.alise = common.default_conv(n_feats, n_feats, 3)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        out = self.reduce(torch.cat([x1, x2], dim=1))
        out = self.alise(out)
        return self.weight1(x) + self.weight2(out)


class DeblurFusion(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, num_features=64,
                 out_put_channels=3, num_steps=2, norm_type=None, num_cfbs=3,
                 n_feats=32, n_blocks=1):
        super(DeblurFusion, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        n_feats = 32
        n_blocks = 1

        self.num_steps = num_steps
        self.num_features = num_features
        self.num_cfbs = num_cfbs

        self.head_conv_ir = head_conv_ir(in_channels, out_channels)
        self.Convfuse1 = convfuse(34, 1)

        self.srir = sr(out_channels, 8)
        self.srvi = sr(out_channels, 8)
        self.srir1 = sr1(32, 8)
        self.srvi1 = sr1(32, 8)
        self.srir2 = sr2(8, 1)
        self.srvi2 = sr2(8, 1)

        # define body module
        modules_bodyir = nn.ModuleList()
        for i in range(0, 1):
            modules_bodyir.append(Un(n_feats=n_feats, wn=wn))

        self.bodyir = nn.Sequential(*modules_bodyir)

    def forward(self, ir, vi):
        # 通道升维到32
        upchannel_ir = self.head_conv_ir(ir)
        upchannel_vi = self.head_conv_ir(vi)
        upchannel_ir1 = upchannel_ir
        upchannel_vi1 = upchannel_vi

        # 超分辨模块 - IR分支
        body_in_ir = []
        for i in range(0, 1):
            upchannel_ir = self.bodyir[i](upchannel_ir)
            body_in_ir.append(upchannel_ir)

        body_out_ir = torch.cat(body_in_ir, 1)
        body_out_ir = torch.sigmoid(body_out_ir)
        body_out_ir1 = self.srir(body_out_ir)

        # 超分辨模块 - VI分支
        body_in_vi = []
        for i in range(0, 1):
            upchannel_vi = self.bodyir[i](upchannel_vi)
            body_in_vi.append(upchannel_vi)

        body_out_vi = torch.cat(body_in_vi, 1)
        body_out_vi = torch.sigmoid(body_out_vi)
        body_out_vi1 = self.srvi(body_out_vi)

        body_out_vi3 = self.srvi1(body_out_vi)
        body_out_ir3 = self.srir1(body_out_ir)
        body_out_vi4 = self.srvi2(body_out_vi3)
        body_out_ir4 = self.srir2(body_out_ir3)

        # 融合模块
        imageoutput = (body_out_ir1, body_out_vi1, body_out_vi3,
                       body_out_ir3, body_out_vi4, body_out_ir4)
        imageoutput = torch.cat(imageoutput, 1)
        imageoutput = self.Convfuse1(imageoutput)

        return imageoutput, body_out_ir4, body_out_vi4