import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torchsummary import summary
# from models.model_wrap import *
from model_wrap import *
# from models.linear_attentionX import LinearAttentionX
# from linear_attentionX import LinearAttentionX
__all__ = ['Res2Net', 'res2net50']


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class VONet_Res2Net(nn.Module):

    def __init__(self, block=Bottle2neck, layers=[3, 4, 6, 3], baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes  = 64
        super(VONet_Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale     = scale
        self.conv1     = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1       = nn.BatchNorm2d(64)
        self.relu      = nn.ReLU(inplace=True)
        self.maxpool   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1    = self._make_layer(block, 64  , layers[0])
        self.layer2    = self._make_layer(block, 128  , layers[1], stride=2)	    

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
        # c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        c4, c5, d1 = 512, 512, 256
        det_h = 65
        nhead = 8              
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa   = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb   = nn.BatchNorm2d(det_h)
        
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa   = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb   = nn.BatchNorm2d(d1)
               
        # self.linear = LinearAttentionX(d_model=d1, nhead=nhead)
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)        
        # x = self.layer3(x)       
        
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))        
                
        # Transformer         
        # desc  = self.linear(cDa)         
        # desc  = self.linear(desc)    
              
        # Norm
        dn = torch.norm(desc, p=2, dim=1)       # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.                           

        output = {'semi': semi, 'desc': desc}

        self.output = output     
        return output     
               
if __name__ == '__main__':   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VONet_Res2Net()
    model = model.to(device)

    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 240, 320))

