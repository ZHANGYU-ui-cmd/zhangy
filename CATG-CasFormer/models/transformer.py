# This file is part of CATGCasFormer, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/CATGCasFormer/blob/master/LICENSE for details.

import math
import random
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.mask import exchange_token, exchange_patch, get_mask_box, jigsaw_token, cutout_patch, erase_patch, mixup_patch, jigsaw_patch
from torchvision.models.shufflenetv2 import channel_shuffle
from torchvision.ops.misc import SqueezeExcitation

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(SEShuffleUnit, self).__init__()
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=self.groups, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.DWconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.SENet = SqueezeExcitation(in_channels, in_channels)
        self.group_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=self.groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        # Group convolution 1
        out = self.group_conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # Channel shuffle
        out = channel_shuffle(out, self.groups)
        # Depthwise convolution
        out = self.DWconv(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.SENet(out)   #不要这个了
        # Group convolution 2
        out = self.group_conv2(out)
        out = self.bn3(out)
        return F.relu(x + out)


#空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

#池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

#ASPP
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels * 4, 1, bias=False),
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(),
            nn.Dropout(0.))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class TransformerHead(nn.Module):
    def __init__(
        self,
        cfg,
        trans_names, 
        kernel_size,
        use_feature_mask,
    ):
        super(TransformerHead, self).__init__()
        d_model = cfg.MODEL.TRANSFORMER.DIM_MODEL   #512

        # Mask parameters
        self.use_feature_mask = use_feature_mask
        mask_shape = cfg.MODEL.MASK_SHAPE    #'stripe'   ‘square’用块
        mask_size = cfg.MODEL.MASK_SIZE   # 1
        mask_mode = cfg.MODEL.MASK_MODE   #'random_direction'

        self.bypass_mask = exchange_patch(mask_shape, mask_size, mask_mode)
        self.get_mask_box = get_mask_box(mask_shape, mask_size, mask_mode)

        self.transformer_encoder = Transformers(
            cfg=cfg,
            trans_names=trans_names, 
            kernel_size=kernel_size,
            use_feature_mask=use_feature_mask,
        )
        
        self.transformer_encoder_2 = Transformers(
            cfg = cfg,
            trans_names = trans_names,
            kernel_size = kernel_size,
            use_feature_mask=False,
        )
        
        self.conv0 = conv1x1(1024, 1024)
        self.conv1 = conv1x1(1024, d_model)
        self.conv2 = conv1x1(d_model, 2048)

        self.final_in = nn.Conv2d(in_channels=384, out_channels=d_model, kernel_size=1)
        self.norm = nn.BatchNorm2d(d_model)
        self.final_in2 = nn.Conv2d(in_channels=d_model, out_channels=1024, kernel_size=1)
        self.final_in3 = nn.Conv2d(1024, d_model, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.seshufflenet = SEShuffleUnit(d_model, d_model, d_model)
        self.conv_aspp = nn.Conv2d(d_model, 256, kernel_size=1)
        self.aspp = ASPP(in_channels=256, atrous_rates=[3, 6, 12], out_channels=256)

        self.conv3 = nn.Conv2d(512, 1024, kernel_size=1)

    def forward(self, box_features):
        #第一个模块
        #卷积-BN-卷积
        final_in = self.final_in(box_features)
        #shuffle_unit  也就是convolutional encoder
        shuffle_feats = self.seshufflenet(final_in)
        shuffle_feats = self.conv_aspp(shuffle_feats)   #256

        #ASPP
        box_features = self.aspp(shuffle_feats)   #通道数为1024

        #遮挡   用块表示
        mask_box = self.get_mask_box(box_features)

        if self.use_feature_mask:
            skip_features = self.conv0(box_features)
            if self.training:
                skip_features = self.bypass_mask(skip_features)
        else:
            skip_features = box_features

        trans_features = {}
        trans_features["before_trans"] = F.adaptive_max_pool2d(skip_features, 1)
        box_features = self.conv1(box_features)     #1024-512

        box_features = self.transformer_encoder((box_features,mask_box))
        box_features = self.transformer_encoder_2((box_features, mask_box))
        box_features = self.conv2(box_features)
        trans_features["after_trans"] = F.adaptive_max_pool2d(box_features, 1)

        return trans_features


class Transformers(nn.Module):
    def __init__(
        self,
        cfg,
        trans_names, 
        kernel_size,
        use_feature_mask,
    ):
        super(Transformers, self).__init__()
        d_model = cfg.MODEL.TRANSFORMER.DIM_MODEL  #512
        self.feature_aug_type = cfg.MODEL.FEATURE_AUG_TYPE   # # 'exchange_token'
        self.use_feature_mask = use_feature_mask    #False

        # If no conv before transformer, we do not use scales
        if not cfg.MODEL.TRANSFORMER.USE_PATCH2VEC:
            trans_names = ['scale1']
            kernel_size = [(1,1)]

        self.trans_names = trans_names   #['scale1','scale2']
        self.scale_size = len(self.trans_names)    #2
        hidden = d_model//(2*self.scale_size)    #128

        # kernel_size: (padding, stride)
        kernels = {
            (1,1): [(0,0),(1,1)],
            (3,3): [(1,1),(1,1)]
        }

        padding = []
        stride = []
        for ksize in kernel_size:    #[(1,1),(3,3)]
            if ksize not in [(1,1),(3,3)]:
                raise ValueError('Undefined kernel size.')
            padding.append(kernels[ksize][0])   #(0,0)  (1,1)
            stride.append(kernels[ksize][1])    #(1,1)  (1,1)

        self.use_output_layer = cfg.MODEL.TRANSFORMER.USE_OUTPUT_LAYER     #False
        self.use_global_shortcut = cfg.MODEL.TRANSFORMER.USE_GLOBAL_SHORTCUT  #True

        self.blocks = nn.ModuleDict()
        for tname, ksize, psize, ssize in zip(self.trans_names, kernel_size, padding, stride):
            transblock = Transformer(
                cfg, d_model//self.scale_size, ksize, psize, ssize, hidden, use_feature_mask
            )   #d_model//self.scale_size: 256,  ksize: [(1,1),(3,3)], psize: #(0,0)  (1,1)  ssize: (1,1)  (1,1)
            self.blocks[tname] = nn.Sequential(transblock)

        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mask_para = [cfg.MODEL.MASK_SHAPE, cfg.MODEL.MASK_SIZE, cfg.MODEL.MASK_MODE]

    def forward(self, inputs):
        trans_feat = []
        enc_feat, mask_box = inputs

        if self.training and self.use_feature_mask and self.feature_aug_type == 'exchange_patch':
            feature_mask = exchange_patch(self.mask_para[0], self.mask_para[1], self.mask_para[2])
            enc_feat = feature_mask(enc_feat)

        for tname, feat in zip(self.trans_names, torch.chunk(enc_feat, len(self.trans_names), dim=1)):
            #trans_name: ['scale1','scale2']  torch.chunk(,dim=1)  在dim=1,也就是第二维，分为2个块
            feat = self.blocks[tname]((feat, mask_box))
            trans_feat.append(feat)

        trans_feat = torch.cat(trans_feat, 1)
        if self.use_output_layer:
            trans_feat = self.output_linear(trans_feat)
        if self.use_global_shortcut:
            trans_feat = enc_feat + trans_feat
        return trans_feat


class Transformer(nn.Module):
    def __init__(self, cfg, channel, kernel_size, padding, stride, hidden, use_feature_mask):
        ##d_model//self.scale_size: 256,  ksize: [(1,1),(3,3)], psize: (0,0)  (1,1)  ssize: (1,1)  (1,1)  hidden: 128
        super(Transformer, self).__init__()
        self.k = kernel_size[0]
        stack_num = cfg.MODEL.TRANSFORMER.ENCODER_LAYERS   #1
        num_head = cfg.MODEL.TRANSFORMER.N_HEAD    #8
        dropout = cfg.MODEL.TRANSFORMER.DROPOUT    #0.
        output_size = (14,14)
        token_size = tuple(map(lambda x,y:x//y, output_size, stride))  #token_size: (14,14)
        blocks = []
        self.transblock = TransformerBlock(token_size, hidden=hidden, num_head=num_head, dropout=dropout)
        for _ in range(stack_num):
            blocks.append(self.transblock)
        self.transformer = nn.Sequential(*blocks)
        self.patch2vec = nn.Conv2d(channel, hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.vec2patch = Vec2Patch(channel, hidden, output_size, kernel_size, stride, padding)
        self.use_local_shortcut = cfg.MODEL.TRANSFORMER.USE_LOCAL_SHORTCUT  #True
        self.use_feature_mask = use_feature_mask
        self.feature_aug_type = cfg.MODEL.FEATURE_AUG_TYPE   ## 'exchange_token'
        self.use_patch2vec = cfg.MODEL.TRANSFORMER.USE_PATCH2VEC

    def forward(self, inputs):
        enc_feat, mask_box = inputs
        b, c, h, w = enc_feat.size()

        trans_feat = self.patch2vec(enc_feat)

        _, c, h, w = trans_feat.size()
        trans_feat = trans_feat.view(b, c, -1).permute(0, 2, 1)

        # For 1x1 & 3x3 kernels, exchange tokens
        if self.training and self.use_feature_mask:
            if self.feature_aug_type == 'exchange_token':
                feature_mask = exchange_token()
                trans_feat = feature_mask(trans_feat, mask_box)
            elif self.feature_aug_type == 'cutout_patch':
                feature_mask = cutout_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'erase_patch':
                feature_mask = erase_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'mixup_patch':
                feature_mask = mixup_patch()
                trans_feat = feature_mask(trans_feat)

        if self.use_feature_mask:
            if self.feature_aug_type == 'jigsaw_patch':
                feature_mask = jigsaw_patch()
                trans_feat = feature_mask(trans_feat)
            elif self.feature_aug_type == 'jigsaw_token':
                feature_mask = jigsaw_token()
                trans_feat = feature_mask(trans_feat)

        trans_feat = self.transformer(trans_feat)
        trans_feat = self.vec2patch(trans_feat)
        if self.use_local_shortcut:
                trans_feat = enc_feat + trans_feat

        return trans_feat


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, tokensize, hidden=128, num_head=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(tokensize, d_model=hidden, head=num_head, p=dropout)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.norm1(x)
        x = x + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)

        return x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class Vec2Patch(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)
        feat = self.to_patch(feat)

        return feat

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, tokensize, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = tokensize

    def forward(self, x):
        b, n, c = x.size() 
        c_h = c // self.head
        key = self.key_embedding(x)
        query = self.query_embedding(x)
        value = self.value_embedding(x)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        att, _ = self.attention(query, key, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        output = self.output_linear(att)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x
