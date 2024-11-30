# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 22:40:36 2022

@author: User
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models._utils import IntermediateLayerGetter

import torchvision

from collections import OrderedDict

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def fusion(xin: Tensor, xup: Tensor, xmid: Tensor, xdown: Tensor):    
    xout = (torch.cat((xup*xmid,xdown*xmid),1))+xin
    return xout

class MSFBlock(nn.Module):
    # MSF Block
    expansion: int = 4

    def __init__(
        self,
        planes: int,
        wxh:int,

        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MSFBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes)
        out_width = int(wxh)

        self.conv1 = conv1x1(width, out_width)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1)) 
        
        self.bn2 = norm_layer(out_width )
        
        self.conv2 = conv1x1(width, out_width)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1)) 
        
        self.bn3 = norm_layer(out_width)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x: Tensor) -> Tensor:
        xOF = x        
        out = self.conv1(x)
        out = self.avgpool1(out)
        out = xOF*out
        
        out = self.bn2(out)
        out = self.relu(out)
        xOF = out

        out = self.conv2(out)
        out = self.avgpool2(out)
        out = xOF*out
        
        out = self.bn3(out)
        out = self.relu(out)
             
        return out

class res_msf(nn.Module):
    
    def __init__(
        self,
        backbon: nn.Module,
        norm_layer: Optional[Callable[..., nn.Module]] = None
     ) -> None:
        super(res_msf, self).__init__()
        self._norm_layer = norm_layer
        return_layers = {"layer2": "upper", "layer3": "middle", "layer4": "bottom"}
        self.imlg = IntermediateLayerGetter(backbon, return_layers) 
        
        # MSF Layer , input 是 layer2 =========================================
        self.upper_layer = MSFBlock(512,512,self._norm_layer)
        self.ul_avgpool = nn.AvgPool2d(2, stride=2)
        #======================================================================
        
        # MSF Layer , input 是 layer3 =========================================
        self.middle_layer = MSFBlock(1024,1024,self._norm_layer)
        self.mid_down_chanel = conv1x1(1024, 512) 
        #=====================================================================
        
        # MSF Layer , input 是 layer4 =========================================
        self.bottom_layer = MSFBlock(2048,2048,self._norm_layer)
        self.bot_up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bot_downchanel = conv1x1(2048, 512)
        #=====================================================================
        
        # fusion =============================================================
        self.fusion = fusion
        #=====================================================================
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: Tensor) -> Tensor:
        
        
        res_layer = self.imlg(x)
            # =============================================================================
            #         {
            #          'upper' : [1, 512, 112, 112]
            #          'middle' : [1, 1024, 56, 56]
            #          'bottom' : [1, 2048, 28, 28]
            #          }
            # =============================================================================
        # MSF upper_layer , input 是 layer2 
        upper_layer = self.upper_layer(res_layer['upper'])                  # upper_layer.shape : [1, 512, 112, 112]
        upper_layer = self.ul_avgpool(upper_layer)                          # upper_layer.shape : [1, 512, 56, 56]
            
        # MSF middle_layer , input 是 layer3 
        middle_layer = self.middle_layer(res_layer['middle'])               # middle_layer.shape : [1, 1024, 56, 56])
        middle_layer = self.mid_down_chanel(middle_layer)                   # middle_layer.shape : [1, 512, 56, 56]
            
        # MSF bottom_layer , input 是 layer4 
        bottom_layer = self.bottom_layer(res_layer['bottom'])               # bottom_layer.shape : [1, 2048, 28, 28]
        bottom_layer = self.bot_up_sample(bottom_layer)                     # bottom_layer.shape : [1, 2048, 56, 56]
        bottom_layer = self.bot_downchanel(bottom_layer)                    # bottom_layer.shape : [1, 512, 56, 56]
        
        
        # fix size not match
        if(bottom_layer.shape[2] != res_layer['middle'].shape[2] or bottom_layer.shape[3] != res_layer['middle'].shape[3]):
            bottom_layer = nn.AdaptiveAvgPool2d((res_layer['middle'].shape[2],res_layer['middle'].shape[3]))(bottom_layer)
        
        if(upper_layer.shape[2] != res_layer['middle'].shape[2] or upper_layer.shape[3] != res_layer['middle'].shape[3]):
            upper_layer = nn.AdaptiveAvgPool2d((res_layer['middle'].shape[2], res_layer['middle'].shape[3]))(upper_layer)
        # =============================================================================
        
        
        # fision =============================================================
        x = self.fusion(res_layer['middle'], upper_layer, middle_layer, bottom_layer) # out.shape : [1, 1024, 56, 56] 
        # =============================================================================
        
        # 為了對接backbone.py===========================================================
        out = OrderedDict()
        out['0'] = x
        # =============================================================================
        
        return out

# 如果要測試的話把這一段註解拿掉 2022/04/13
# =============================================================================
# import sys 
# sys.path.append("..") 
# from util.misc import NestedTensor, is_main_process
# class FrozenBatchNorm2d(torch.nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.
# 
#     Copy-paste from torchvision.misc.ops with added eps before rqsrt,
#     without which any other models than torchvision.models.resnet[18,34,50,101]
#     produce nans.
#     """
# 
#     def __init__(self, n):
#         super(FrozenBatchNorm2d, self).__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))
# 
#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]
# 
#         super(FrozenBatchNorm2d, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)
# 
#     def forward(self, x):
#         # move reshapes to the beginning
#         # to make it fuser-friendly
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         eps = 1e-5
#         scale = w * (rv + eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias
# 
# if __name__ == '__main__':
#     backbone = getattr(torchvision.models, 'resnet50')(
#             replace_stride_with_dilation=[False, False, False],
#             pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
#     # X = torch.randn((1,3,896,896))
#     #X2 = torch.randn((2, 3, 544, 815))
#     X2 = torch.randn((1,3,896,896))
#     #X2 = torch.randn((2, 3, 768, 1151))
#     #X2 = torch.randn((2, 3, 815, 544))
#     #X2 = torch.randn((2, 3, 34, 51))
#     model = res_msf(backbone)
#     out = model(X2)
#     print(model)
#     print(f"out shape : { out['0'].shape }")  # out.shape:[1, 1024, 56, 56]
# =============================================================================
        