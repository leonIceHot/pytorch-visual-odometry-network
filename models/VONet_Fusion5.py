import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# from models.model_wrap import *
from model_wrap import *
# from models.linear_attentionX import LinearAttentionX
from linear_attentionX import LinearAttentionX
# from models.res2net import Bottle2neck  
from res2net import Bottle2neck  


class VONet_Fusion5(nn.Module):

    def __init__(self, block=Bottle2neck, layers=[3, 4, 5], baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes  = 32
        super(VONet_Fusion5, self).__init__()
        
        c0, c1, c2, c3, d1 = 32, 64, 128, 256, 256       
        det_h = 65
        nhead = 8          
        
        # input-convolution  ( H, W, C) -> ( 30, 40, 128)
        self.baseWidth = baseWidth
        self.scale     = scale
        self.conv0     = nn.Conv2d(1, c0, kernel_size=7, stride=2, padding=3, bias=False) # c1 (1 -> 32)
        self.bn1       = nn.BatchNorm2d(c0)                                               # c1 : 32                                           
        self.relu      = nn.ReLU(inplace=True)
        self.maxpool   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1    = self._make_layer(block, c0, layers[0])                           # c1 : 32
        self.layer2    = self._make_layer(block, c1, layers[1], stride=2)                 # c1 -> c2 (32 -> 64)    
        self.layer3    = self._make_layer(block, c2, layers[2], stride=2)                 # c2 -> c3 (64 -> 128)   
                    
        
        # upper-layer ( H, W, C) -> ( 30, 40, 128)
        self.upsampling= nn.Upsample(scale_factor=2, mode='bicubic',align_corners=True) # (2H,2W)
        
        self.conv1 = nn.Sequential(      
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),   # c2 -> c2 ( 128 -> 128)      
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),                            
            Bottle2neck(inplanes=c2, planes=c2),                     # c2 -> c2 ( 128 -> 128)      
            
            nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1),   # c2 -> c1 ( 128 -> 64)
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),                            
            Bottle2neck(inplanes=c1, planes=c1),                     # c1 -> c1 ( 64 -> 64)      

            nn.Conv2d(c1, c0, kernel_size=3, stride=1, padding=1),   # c1 -> c0 ( 64 -> 32)
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True))                                                    
                
        # input-convolution  ( H, W, C) -> ( 30, 40, 128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1),   # c2 -> c1 (128 -> 64)
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(c1, c0, kernel_size=3, stride=1, padding=1),   # c1 -> c0 (64 -> 32)
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True))         
                               
        # down-layer ( H, W, C) -> ( 30, 40, 128)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)           # (H/2, W/2) (15,20)        
        
        self.conv3 = nn.Sequential(                                  
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),   # c2 -> c2 (128 -> 128)
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),             
            Bottle2neck(inplanes=c2, planes=c2),                     # c2 -> c2 (128 -> 128)
             
            nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1),   # c2 -> c1 (128 -> 64)
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            Bottle2neck(inplanes=c1, planes=c1),                     # c1 -> c1 (64 -> 64)             
            
            nn.Conv2d(c1, c0, kernel_size=3, stride=1, padding=1),   # c1 -> c0 (64 -> 32)
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True))            
            
        # others
        self.relu         = nn.ReLU(inplace=True)
        self.conv_fusion  = nn.Conv2d(c0+c0+c0, c1, kernel_size=1, stride=1, padding=0) # c0+c0+c0 -> c1 (96 -> 64)                           
        self.upsampling   = nn.Upsample(scale_factor=2, mode='bicubic',align_corners=True)
        self.downsampling = nn.AvgPool2d(kernel_size=2, stride=2)                          
        self.linear = LinearAttentionX(d_model=c1, nhead=nhead)       
                     
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c1, c3, kernel_size=3, stride=1, padding=1)      # c1 -> c3 (64 -> 256)
        self.bnPa   = nn.BatchNorm2d(c3)
        self.convPb = torch.nn.Conv2d(c3, det_h, kernel_size=1, stride=1, padding=0)   # c3 -> det_h (256 -> 65) 
        self.bnPb   = nn.BatchNorm2d(det_h)
        
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)      # c1 -> c3 (64 -> 128)
        self.bnDa   = nn.BatchNorm2d(c2)
        self.convDb = torch.nn.Conv2d(c2, d1, kernel_size=1, stride=1, padding=0)      # c3 -> d1 (128 -> 256)
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
        
        # Res2Net
        x  = self.conv0(x)
        x  = self.bn1(x)
        x  = self.relu(x)
        x  = self.maxpool(x)
        
        x  = self.layer1(x)
        x  = self.layer2(x) 
        x0  = self.layer3(x) 
        
                             
        # upper-layer           
        x1 = self.upsampling(x0)                      # x:[2,128,60,80]        
        x1 = self.conv1(x1)                           # x:[2,32,30,40]
        downsample = self.downsampling(x1)            # downsample:[2,32,30,40]               
        
        
        # input-convolution    
        x2 = self.conv2(x0)                           # x:[2,32,30,40]               
        
                    
        # down-layer
        x3 = self.pool3(x0)                           # x:[2,128,15,20]                              
        x3 = self.conv3(x3)                           # x:[2,32,15,20]        
        upsample = self.upsampling(x3)                # upsample:[2,32,30,40]
            
        
        # Multi-scale fusion for layers x2, x3, and x4
        fusion_layers = torch.cat((downsample, x2, upsample), dim=1)       # fusion:[2,96,30,40]
        x4            = self.conv_fusion(fusion_layers)                    # x:[2,64,30,40]        
        
             
        # Linear Attention
        x5 = self.linear(x4)       # [-1,64,30,40]
               
        
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x5)))
        semi = self.bnPb(self.convPb(cPa))
        
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x5)))
        desc = self.bnDb(self.convDb(cDa))   
        
        # Channel-wise L2 normalization (L2-norm)
        dn = torch.norm(desc, p=2, dim=1)       # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.  
                     
        output = {'semi': semi, 'desc': desc}

        self.output = output     
        return output          
                                      
if __name__ == '__main__':   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VONet_Fusion5()
    model = model.to(device)

    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 480, 640))

