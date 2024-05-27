import torch
from torch import nn
import torch.nn.functional as F
import math


class SENet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

###improve SENet——sequeeze  FcaNet

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)

def get_dct_weights( width, height, channel, fidx_u= [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3], fidx_v= [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]):
    # width : width of input
    # height : height of input
    # channel : channel of input
    # fidx_u : horizontal indices of selected fequency
    # according to the paper, should be [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3]
    # fidx_v : vertical indices of selected fequency
    # according to the paper, should be [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]
    # [0,0],[0,1],[6,0],[0,5],[0,2],[1,0],[1,2],[4,0],
    # [5,0],[1,6],[3,0],[0,4],[0,6],[0,3],[2,2],[3,5],
    scale_ratio = width//7
    fidx_u = [u*scale_ratio for u in fidx_u]
    fidx_v = [v*scale_ratio for v in fidx_v]
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    # split channel for multi-spectal attention
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i * c_part: (i+1)*c_part, t_x, t_y]\
                =get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    # Eq. 7 in our paper
    return dct_weights


class FcaNet(nn.Module):
    def __init__(self,
                 channel,
                 reduction,width,height):
        super(FcaNet, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights',get_dct_weights(self.width,self.height,channel))
        #self.register_parameter('pre_computed_dct_weights',torch.nn.Parameter(get_dct_weights(width,height,channel)))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x,(self.height,self.width))
        y = torch.sum(y*self.pre_computed_dct_weights,dim=(2,3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


####improved excitation——ECANet
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



## improved SENet——SRM
class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)

#####GCT  Gated channel transformation for visual recognition,
class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


###通道-空间注意力机制

##CBAM    https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py   https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
##2018
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(in_planes, ratio)
        self.SpatialAttention = SpatialAttention()

    def forward(self, x):
        x_out = self.ChannelAttention(x)
        x_out = self.SpatialAttention(x_out)
        return x * x_out.expand_as(x)
###上面这个通常都是放到resnet50中的并与x相加


##BAM  2018
class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i+1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i+1), nn.BatchNorm1d(gate_channels[i+1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i+1), nn.ReLU())
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel // reduction_ratio, kernel_size=3,padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel//reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, x):
        att = 1 + F.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att * x.expand_as(x)


####triplet attention  2021   由三个平行的分支组成，两个负责捕捉通道维度Ch和空间维度H或W之间的跨维度相互作用，剩余的最后一个分支类似于CBAM建立空间注意，所有的三分之一的输出通过简单的平均进行汇总

class channelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class spatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = channelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = spatialGate()
        self.ChannelGateW = spatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out



##SimAM

class simam_module(torch.nn.Module):

    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

### coordinate attention  将位置信息嵌入到通道注意力中为移动网络提出一种新的注意力机制——坐标注意力   2021
#坐标注意力将通道注意力分解为两个一维特征编码过程，分别沿两个空间方向聚合特征。
# 通过这种方式，可以沿一个空间方向捕获远程依赖关系，同时可以沿另一个空间方向保留精确的位置信息。
# 然后将得到的特征图单独编码成一对方向感知和位置敏感的注意力图，这些图可以互补地应用于输入特征图以增强感兴趣对象的表示。
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

##RGA  2020

