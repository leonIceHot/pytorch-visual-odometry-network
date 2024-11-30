import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from torchsummary import summary
from torchvision.ops import roi_pool
import torchgeometry as tgm
# from models.res2net import Bottle2neck                                 # 2020/02/19
from res2net import Bottle2neck                                      # 2020/02/19

class VONet_Fusion2(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, conf_thresh=0.015, nms_dist=4, patch_size=5):
        super(VONet_Fusion2, self).__init__()
        self.conf_thresh    = conf_thresh
        self.nms_dist       = nms_dist
        self.patch_size     = patch_size
        self.out_num_points = 500
        self.output         = None
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        c0, c1, c2, c3, c4, c5, d1 = 32, 64, 64, 128, 128, 256, 256
        det_h = 65
        # input-convolution
        self.conv0 = nn.Sequential(            
            nn.Conv2d(1, c0, kernel_size=3, stride=1, padding=1),    # 1 -> c0 (1 -> 32)
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True), 
            
            Bottle2neck(inplanes=c0, planes=c0),                     # c0 -> c0 (32 -> 32)                                                                                                                                         
            Bottle2neck(inplanes=c0, planes=c0)                      # c0 -> c0 (32 -> 32)                                  
        )
        # downsize-1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/2, W/2)
        self.conv1 = nn.Sequential(                                  # 2022/03/27
            Bottle2neck(inplanes=c0, planes=c1),                     # c0 -> c1 (32 -> 64)
            Bottle2neck(inplanes=c1, planes=c1)                      # c1 -> c1 (64 -> 64)
        )

        # downsize-2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/4, W/4)
        self.conv2 = nn.Sequential(                                  # 2022/03/27
             Bottle2neck(inplanes=c2, planes=c3),                    # c2 -> c3 (64 -> 128)
             Bottle2neck(inplanes=c3, planes=c3)                     # c3 -> c3 (128 -> 128)
            )

        # downsize-3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/8, W/8)
        self.conv3 = nn.Sequential(                                  # 2022/03/27
             Bottle2neck(inplanes=c3, planes=c4),                    # c3 -> c4 (128 -> 128)
             Bottle2neck(inplanes=c4, planes=c4)                     # c4 -> c4 (128 -> 128)
            )

        # downsize-4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/16, W/16)
        self.conv4 = nn.Sequential(                                  # 2022/03/27
             Bottle2neck(inplanes=c4, planes=c5),                    # c4 -> c5 (128 -> 256)
             Bottle2neck(inplanes=c5, planes=c5)                     # c5 -> c5 (256 -> 256)
            )

        # others
        self.relu         = nn.ReLU(inplace=True)
        self.conv_fusion  = nn.Conv2d(c3+c4+c5, c3, kernel_size=1, stride=1, padding=0) # c3+c4+c5 -> c3 (512 -> 128)
        
        # self.upsampling   = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsampling   = nn.Upsample(scale_factor=2, mode='bicubic',align_corners=True)
        self.downsampling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)    # c4 -> c5 (128 -> 256)
        self.bnPa   = nn.BatchNorm2d(c5)
        self.convPb = nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0) # c5 -> det_h (256 -> 65) 
        self.bnPb   = nn.BatchNorm2d(det_h)
        
        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)    # c4 -> c5 (128 -> 256)
        self.bnDa   = nn.BatchNorm2d(c5)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)    # c5 -> d1 (256 -> 256)
        self.bnDb   = nn.BatchNorm2d(d1)
        

    def forward(self, x):                             # x:[2,1,240,320]
        """ Forward pass that jointly computes unprocessed point and descriptor tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # input-convolution
        x = self.conv0(x)                           # x:[2,64,240,320]
        
        # downsize-1
        x = self.pool1(x)                           # x:[2,64,120,160]
        x = self.conv1(x)                           # x:[2,64,120,160]
        
        # downsize-2
        x = self.pool2(x)                           # x:[2,64,60,80]
        x = self.conv2(x)                           # x:[2,128,60,80]
        downsample = self.downsampling(x)           # downsample:[2,128,30,40]
        
        # downsize-3
        x  = self.pool3(x)                          # x:[2,128,30,40]
        x  = self.conv3(x)                          # x:[2,128,30,40]
        x3 = x
        
        # downsize-4
        x = self.pool4(x)                           # x:[2,128,15,20]
        x = self.conv4(x)                           # x:[2,256,15,20]
        upsample = self.upsampling(x)               # upsample:[2,256,30,40]
        
        # Multi-scale fusion for layers x2, x3, and x4
        # fusion_layers = torch.cat((downsample, x3, upsample), dim=1)       # fusion:[2,512,30,40]
        fusion_layers = torch.cat((downsample, x3, upsample), dim=1)       # fusion:[2,512,30,40]
        x3            = self.conv_fusion(fusion_layers)                    # x:[2,128,30,40]
        
        # Detector Head.
        cPa  = self.relu(self.bnPa(self.convPa(x3)))  # cPa:[2,256,30,40]
        semi = self.bnPb(self.convPb(cPa))            # semi:[2,65,30,40]
        
        # Descriptor Head.
        cDa  = self.relu(self.bnDa(self.convDa(x3)))  # cDa:[2,256,30,40]
        desc = self.bnDb(self.convDb(cDa))            # desc:[2,256,30,40]
        
        # Channel-wise L2 normalization (L2-norm)
        dn   = torch.norm(desc, p=2, dim=1)           # dn:[2,30,40]
        desc = desc.div(torch.unsqueeze(dn, 1))       # torch.unsqueeze(dn,1):[2,256,30,40]
        
        # output results from the backbone
        output      = {'semi': semi, 'desc': desc}    # output['semi']:[2,65,30,40] 
        self.output = output
                                
        return output

if __name__ == '__main__':   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VONet_Fusion2()
    model = model.to(device)

    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 240, 320))



