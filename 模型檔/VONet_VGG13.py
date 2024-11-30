from torchsummary import summary
import torch
import torch.nn as nn

class VONet_VGG13(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, conf_thresh=0.015, nms_dist=4, patch_size=5):
        super(VONet_VGG13, self).__init__()
        self.conf_thresh    = conf_thresh
        self.nms_dist       = nms_dist
        self.patch_size     = patch_size
        self.out_num_points = 500
        self.output         = None
        self.device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        # input-convolution
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1),    # 1 -> c1 (1 -> 64)
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1),   # c1 -> c1 (64 -> 64)
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        # downsize-1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/2, W/2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),   # c1 -> c2 (64 -> 64)
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),   # c2 -> c2 (64 -> 64)
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

        # downsize-2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/4, W/4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),   # c2 -> c3 (64 -> 128)
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1),   # c3 -> c3 (128 -> 128)
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

        # downsize-3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/8, W/8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1),   # c3 -> c4 (128 -> 128)
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1),   # c4 -> c4 (128 -> 128)
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)    # c4 -> c5 (128 -> 256)
        self.bnPa   = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0) # c5 -> det_h (256 -> 65) 
        self.bnPb   = nn.BatchNorm2d(det_h)
        
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)    # c4 -> c5 (128 -> 256)
        self.bnDa   = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)    # c5 -> d1 (256 -> 256)
        self.bnDb   = nn.BatchNorm2d(d1)
        

    def forward(self, x):                             # x:[2,1,120,160]
        """ Forward pass that jointly computes unprocessed point and descriptor tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # input-convolution
        x0 = self.conv0(x)                            # x0:[2,64,120,160]
        
        # downsize-1
        x1 = self.pool1(x0)                           # x1:[2,64,60,80]
        x1 = self.conv1(x1)                           # x1:[2,64,60,80]
        
        # downsize-2
        x2 = self.pool2(x1)                           # x2:[2,64,30,40]
        x2 = self.conv2(x2)                           # x2:[2,128,30,40]
        
        # downsize-3
        x3 = self.pool3(x2)                           # x3:[2,128,15,20]
        x3 = self.conv3(x3)                           # x3:[2,128,15,20]
        
        # Detector Head.
        cPa  = self.relu(self.bnPa(self.convPa(x3)))  # cPa:[2,256,15,20]
        semi = self.bnPb(self.convPb(cPa))            # semi:[2,65,15,20]
        
        # Descriptor Head.
        cDa  = self.relu(self.bnDa(self.convDa(x3)))  # cDa:[2,256,15,20]
        desc = self.bnDb(self.convDb(cDa))            # desc:[2,256,15,20]
        
        # Channel-wise L2 normalization (L2-norm)
        dn   = torch.norm(desc, p=2, dim=1)           # dn:[2,15,20]
        desc = desc.div(torch.unsqueeze(dn, 1))       # torch.unsqueeze(dn,1):[2,1,15,20]
        
        # output results from the backbone
        output      = {'semi': semi, 'desc': desc}    # output['semi']:[2,65,15,20] 
        self.output = output

        return output

if __name__ == '__main__':   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = VONet_VGG13()
    model  = model.to(device)
    summary(model, input_size=(1, 240, 320))



