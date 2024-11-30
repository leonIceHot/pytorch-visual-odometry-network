import torch.nn as nn
import math
import torch
from torchsummary import summary

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ChannelAttention(nn.Module):
    
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :   
        max_result = self.maxpool(x)   # [2, 128, 1, 1]
        avg_result = self.avgpool(x)   # [2, 128, 1, 1]
        max_out = self.se(max_result)  # [2, 128, 1, 1]
        avg_out = self.se(avg_result)  # [2, 128, 1, 1]
        output = self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_ = torch.max(x,dim=1,keepdim=True)  # [2, 1, 30, 40]
        avg_result = torch.mean(x,dim=1,keepdim=True)   # [2, 1, 30, 40]
        result = torch.cat([max_result,avg_result],1)   # [2, 2, 30, 40]
        output = self.conv(result)                      # [2, 1, 30, 40]
        output = self.sigmoid(output)                   # [2, 1, 30, 40]
        return output
    
class CSP_CBAM_Block(nn.Module):

    def __init__(self, channel,reduction,kernel_size):
        """ Constructor
        Args:
            channel: input channel dimensionality for cbam
            channel2: input channel dimensionality for dimension reduction or dimension raising
        """
        super().__init__()
        
        c2 = torch.div(channel,2,rounding_mode='floor')  
        self.conv4 = nn.Conv2d(channel//2,channel,1,bias=False)   # 128 -> 256        
        self.ca = ChannelAttention(channel=c2,reduction=reduction)        
        self.sa = SpatialAttention(kernel_size=kernel_size)                              
        
    def forward(self, x):                              
        x = x              
        residual = x                              # [-1, 256, 30, 40]
                                            
        b, c, _, _ = x.size()                     # [-1, 256, 30, 40]    
        c2 = torch.floor_divide(c,2)          
        
        # upper-layer                       
        xu   = x[:,:c2,:,:] 
        out  = xu  * self.ca(xu)                  # [-1, 128, 30, 40]                                 
        out  = out * self.sa(out)                 # [-1, 128, 30, 40]        
        out1 = self.conv4(out)                    # [-1, 256, 30, 40]
        
        # down-layer         
        xl   = x[:,c2:,:,:]           
        out2 = self.conv4(xl)                     # [-1, 256, 30, 40]                  
        
        # Residual 
        xout = out1 + out2 + residual             # [-1, 256, 30, 40]        
        return xout
    
class Bottle2neck2(nn.Module):
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
        super(Bottle2neck2, self).__init__()

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
          convs.append(CSP_CBAM_Block(channel=width,reduction=13,kernel_size=3))          
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
        residual = x               # [-1, 64,  60, 80]

        out = self.conv1(x)        # [-1, 104, 60, 80]
        out = self.bn1(out)        # [-1, 104, 60, 80]
        out = self.relu(out)       # [-1, 104, 60, 80]

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
    
class VONet_Res2Net_CSP_CBAM2(nn.Module):

    def __init__(self, block=Bottle2neck2,                  
                 layers=[3, 4, 6, 3], baseWidth = 26, scale = 4,
                 num_classes=1000):        
        
        # c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        c1, c2, c3, c4, c5, d1 = 64, 128, 256, 512, 2048, 256        
        det_h = 65             
        
        self.inplanes  = c1
        super(VONet_Res2Net_CSP_CBAM2, self).__init__()
        self.baseWidth = baseWidth
        self.scale     = scale        
        
        self.conv1     = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False)        
        self.bn1       = nn.BatchNorm2d(c1)
        self.relu      = nn.ReLU(inplace=True)        
        self.maxpool   = nn.MaxPool2d(kernel_size=2, stride=2)
                
        self.layer1    = self._make_layer(block, c1 , layers[0]) 
        self.layer2    = self._make_layer(block, c2 , layers[1])
        self.layer3    = self._make_layer(block, c3 , layers[2])
        self.layer4    = self._make_layer(block, c4 , layers[3])        
        self.conv2     = conv1x1(c5, c3)        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)   
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)        
                
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bnPa   = nn.BatchNorm2d(c3)
        self.convPb = torch.nn.Conv2d(c3, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb   = nn.BatchNorm2d(det_h)
        
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bnDa   = nn.BatchNorm2d(c3)
        self.convDb = torch.nn.Conv2d(c3, d1, kernel_size=1, stride=1, padding=0)
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
        x = self.conv1(x)             # [-1, 64, 240, 320]                        
        x = self.bn1(x)               # [-1, 64, 240, 320]                
        x = self.relu(x)              # [-1, 64, 240, 320]                                 
        
        # Stage1                
        x = self.maxpool(x)           # [-1, 64, 120, 160]                  
        x = self.layer1(x)            # [-1, 256, 120, 160]                                                   
        
        # Stage2
        x = self.maxpool(x)           # [-1, 256, 60,  80]             
        x = self.layer2(x)            # [-1, 512, 60,  80] 
        
        # Stage3
        x = self.maxpool(x)           # [-1,  512, 30, 40]            
        x = self.layer3(x)            # [-1, 1024, 30, 40]                   

        # Stage4                            
        x = self.layer4(x)            # [-1, 2048, 30, 40]                                                                                   
        x = self.conv2(x)             # [-1,  256, 30, 40]    
                     
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))        
                             
        # Norm
        dn = torch.norm(desc, p=2, dim=1)       # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.                           

        output = {'semi': semi, 'desc': desc}
        self.output = output     
        return output     
    

if __name__ == '__main__':   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = VONet_Res2Net_CSP_CBAM2()
    model  = model.to(device)
    summary(model, input_size=(1, 240, 320))
