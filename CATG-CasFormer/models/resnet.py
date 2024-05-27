from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
import torch
from typing import Dict
from torch import Tensor

class OldBackbone(nn.Sequential):
    def __init__(self, resnet):
        super(OldBackbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(OldBackbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class OldRes5Head(nn.Sequential):
    def __init__(self, resnet):
        super(OldRes5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(OldRes5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return OldBackbone(resnet), OldRes5Head(resnet)


class Backbone(nn.Module):
    def forward(self, x):
        y = self.body(x)
        return y


class ConvnextBackbone(Backbone):
    def __init__(self, convnext):
        super().__init__()
        return_layers = {
            '5': 'feat_res4',
        }
        self.body = IntermediateLayerGetter(convnext.features, return_layers=return_layers)
        self.out_channels = convnext.features[5][-1].block[5].out_features


# Generic Head
class Head(nn.Module):
    def forward(self, x) -> Dict[str, Tensor]:
        feat = self.head(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": x, "feat_res5": feat}

class ConvnextHead(Head):
    def __init__(self, convnext):
        super().__init__()
        self.head = nn.Sequential(
            convnext.features[6],
            convnext.features[7],
        )
        self.out_channels = [
            convnext.features[5][-1].block[5].out_features,
            convnext.features[7][-1].block[5].out_features,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']


# convnext model builder function
def build_convnext(arch='convnext_base', pretrained=True, freeze_layer1=True):
    # weights
    weights = None

    # load model
    if arch == 'convnext_tiny':
        print('==> Backbone: ConvNext Tiny')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_tiny(weights=weights)
    elif arch == 'convnext_small':
        print('==> Backbone: ConvNext Small')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_small(weights=weights)
    elif arch == 'convnext_base':
        print('==> Backbone: ConvNext Base')
        if pretrained:
            weights = torchvision.models.convnext.__dict__[arch](pretrained=pretrained)
#            weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_base(weights=weights)
    elif arch == 'convnext_large':
        print('==> Backbone: ConvNext Large')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_large(weights=weights)
    else:
        raise NotImplementedError

    # freeze first layer
    if freeze_layer1:
        convnext.features[0].requires_grad_(False)

    # setup backbone architecture
    backbone, head = ConvnextBackbone(convnext), ConvnextHead(convnext)

    # return backbone, head
    return backbone, head