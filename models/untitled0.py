import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import Tensor
from torchsummary import summary
# from models.linear_attentionX import LinearAttentionX
from linear_attentionX import LinearAttentionX
# from models.res2net import Bottle2neck,Bottle2neck2  
from res2net import Bottle2neck
from typing import Callable, Optional

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def fusion(xin:Tensor,xup:Tensor,xmid:Tensor,xdown:Tensor):
    xout = (torch.cat((xup*xmid,xdown*xmid),1)) + xin
    return xout

class MSFBlock(nn.Module):
    # MSF Block
    expansion: int = 4
    
    def __init__(
        self,
        planes: int,
        wxh:int,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        
        super(MSFBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes)
        out_width = int(wxh)

        self.conv1 = conv1x1(width, out_width)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))         
        self.bn2 = norm_layer(out_width)
        
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
    
class VONet_Fusion4(nn.Module):

    def __init__(self, 
                 block=Bottle2neck, 
                 layers=[3, 4], 
                 baseWidth = 26, 
                 scale = 8, 
                 num_classes=1000,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        
        self.inplanes  = 64
        super(VONet_Fusion4, self).__init__()
        self.norm_layer = norm_layer
        
        c1, c2, c3, c4, c5, d1 = 64, 128, 256, 512, 1024, 256        
        det_h = 65
        nhead = 8          
        
        # input-convolution  ( H, W, C) -> ( 30, 40, 128)
        self.baseWidth = baseWidth
        self.scale     = scale
        self.conv0     = nn.Conv2d(1, c1, kernel_size=7, stride=2, padding=3, bias=False) # c1 (1 -> 64)
        self.bn1       = nn.BatchNorm2d(c1)                                               # c1 : 64                                           
        self.relu      = nn.ReLU(inplace=True)
        self.maxpool   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1    = self._make_layer(block, c1, layers[0])                           # c1 (64 -> 256)
        self.layer2    = self._make_layer(block, c2, layers[1], stride=2)                 # c2 (128-> 512)        
        self.conv      = conv1x1(c4,c3)
        
        # upper-layer ( H, W, C) -> ( 30, 40, 512)
        self.upsample1 = nn.Sequential(
             nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True), # (2H,2W)    (60 -> 80) 
             conv1x1(c3,c2))
             
        self.upperlayer = MSFBlock(c2,c2,self.norm_layer)         # c4 -> c3 ( 512 -> 256)
        self.downsample2 = nn.Sequential(
             nn.AvgPool2d(kernel_size=2, stride=2),
             conv1x1(c2,c3))  # c3 -> c3 ( 256 -> 256) 
                   
        # middle-layer  ( H, W, C) -> ( 30, 40, 512)
        self.middlelayer = nn.Sequential(
            conv1x1(c2,c3),
            MSFBlock(c3,c3,self.norm_layer))        # c3 -> c3   (256 -> 256) 
                   
        # down-layer ( H, W, C) -> ( 30, 40, 512)
        self.downsample1 = nn.Sequential(
             nn.AvgPool2d(kernel_size=2, stride=2),               # (H/2, W/2) (15,20) 
             conv1x1(c3,c4))               
               
        self.downlayer = MSFBlock(c4,c4,self.norm_layer)                
        self.upsample2 = nn.Sequential(                           # c3 -> c3 ( 256 -> 256) 
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True), # c5 -> c5   (1024 -> 1024)
            conv1x1(c4,c3))
                                                
        # Fusion          
        self.fusion = fusion
                                                          
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c3, kernel_size=3, stride=1, padding=1)      # c5 -> c3 (1024 -> 256)
        self.bnPa   = nn.BatchNorm2d(c3)
        self.convPb = torch.nn.Conv2d(c3, det_h, kernel_size=1, stride=1, padding=0)   # c3 -> det_h (256 -> 65) 
        self.bnPb   = nn.BatchNorm2d(det_h)
        
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c3, kernel_size=3, stride=1, padding=1)      # c4 -> c3 (512 -> 256)
        self.bnDa   = nn.BatchNorm2d(c3)
        self.convDb = torch.nn.Conv2d(c3, d1, kernel_size=1, stride=1, padding=0)      # c3 -> d1 (256 -> 256)
        self.bnDb   = nn.BatchNorm2d(d1)               
        
        self.output = None        
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        
        # Res2Net stage2
        x  = self.conv0(x)
        x  = self.bn1(x)
        x  = self.relu(x)
        x  = self.maxpool(x)
                
        x  = self.layer1(x)        
        xin1 = self.layer2(x)                       # x:[-1,512,30,40]   
        
        # Dimension Reduction
        xin  = self.conv(xin1)                      # x:[-1,256,30,40]
                          
        # upper-layer                   
        x1 = self.upsample1(xin)                    # x:[-1,128,60,80]                                                                                  
        x1 = self.upperlayer(x1)                    # x:[-1,128,60,80]                     
        
        # middle-layer            
        x2 = self.middlelayer(xin)                  # x:[-1,256,30,40]                                           
                
        # down-layer
        x3 = self.downsample1(xin)                  # x:[-1,512,15,20]                                                                                                  
        x3 = self.downlayer(x3)                     # x:[-1,512,15,20]                                                     
        
        # Multi-scale fusion 
        downsample = self.downsample2(x1)           # x1:[-1,128,30,40]                                                        
        upsample   = self.upsample2(x3)             # x3:[-1,128,30,40]                                                                                          
        x4         = self.fusion(xin1,downsample,x2,upsample)         
        
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))        
        semi = self.bnPb(self.convPb(cPa))
        
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))   
        
        # Channel-wise L2 normalization (L2-norm)
        dn = torch.norm(desc, p=2, dim=1)       # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.  
                     
        output = {'semi': semi, 'desc': desc}
        
        self.output = output             
        return output    
        
                            
if __name__ == '__main__':   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VONet_Fusion4()
    model = model.to(device)

    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 240, 320))

