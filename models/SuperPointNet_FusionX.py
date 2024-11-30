import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from torchsummary import summary
from torchvision.ops import roi_pool
import torchgeometry as tgm
from res2net import Bottle2neck                                      # 2020/02/19

class SuperPointNetX(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, conf_thresh=0.015, nms_dist=4, patch_size=5):
        super(SuperPointNetX, self).__init__()
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
        self.conv1 = nn.Sequential(                                  # 2020/02/19
            Bottle2neck(inplanes=c1, planes=c2),                     # c1 -> c2 (64 -> 64)
            Bottle2neck(inplanes=c2, planes=c2)                      # c2 -> c2 (64 -> 64)
        )

        # downsize-2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/4, W/4)
        self.conv2 = nn.Sequential(                                  # 2022/02/19
             Bottle2neck(inplanes=c2, planes=c3),                    # c2 -> c3 (64 -> 128)
             Bottle2neck(inplanes=c3, planes=c3)                     # c3 -> c3 (128 -> 128)
            )

        # downsize-3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/8, W/8)
        self.conv3 = nn.Sequential(                                  # 2022/02/19
             Bottle2neck(inplanes=c3, planes=c4),                    # c3 -> c4 (128 -> 128)
             Bottle2neck(inplanes=c4, planes=c4)                     # c4 -> c4 (128 -> 128)
            )

        # downsize-4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)           # (H/16, W/16)
        self.conv4 = nn.Sequential(                                  # 2022/02/19
             Bottle2neck(inplanes=c4, planes=c5),                    # c4 -> c5 (128 -> 256)
             Bottle2neck(inplanes=c5, planes=c5)                     # c5 -> c5 (256 -> 256)
            )

        # others
        self.relu = nn.ReLU(inplace=True)
        self.conv_fusion = torch.nn.Conv2d(c3+c4+c5, c3, kernel_size=1, stride=1, padding=0) # c3+c4+c5 -> c3 (512 -> 128)
        
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
        

    def forward(self, x):                             # x:[2,1,240,320]
        """ Forward pass that jointly computes unprocessed point and descriptor tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # input-convolution
        x0 = self.conv0(x)                            # x0:[2,64,240,320]
        # downsize-1
        x1 = self.pool1(x0)                           # x1:[2,64,120,160]
        x1 = self.conv1(x1)                           # x1:[2,64,120,160]
        # downsize-2
        x2 = self.pool2(x1)                           # x2:[2,64,60,80]
        x2 = self.conv2(x2)                           # x2:[2,128,60,80]
        # downsize-3
        x3 = self.pool3(x2)                           # x3:[2,128,30,40]
        x3 = self.conv3(x3)                           # x3:[2,128,30,40]
        # downsize-4
        x4 = self.pool4(x3)                           # x4:[2,128,15,20]
        x4 = self.conv4(x4)                           # x4:[2,256,15,20]

        # Multi-scale fusion for layers x2, x3, and x4
        downsample    = nn.MaxPool2d(kernel_size=2, stride=2)(x2)          # downsample:[2,128,30,40]
        upsample      = nn.Upsample(scale_factor=2, mode='bilinear')(x4)   # upsample:[2,256,30,40]
        fusion_layers = torch.cat((downsample, x3, upsample), dim=1)       # fusion:[2,512,30,40]
        x3            = self.conv_fusion(fusion_layers)
        
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
        
        # output results from descriptor decoder
        outs   = self.process_outputX()
        # outs['pts_int']:[2,500,2]    記錄特徵點(x,y)座標
        # outs['pts_offset']:[2,500,2] 記錄5x5區塊質心偏移量(roi-pooling:7x7->5x5)
        # outs['pts_desc']:[2,500,256] 記錄特徵點256維的描述子
        
        output.update(outs)
        # output.update({'semi':semi, 'desc':desc, 
        #    'pts_int':pts_int, 'pts_offset':pts_offset, 'pts_desc':pts_desc})
        self.output = output                          
        
        return output

    def heatmap_nmsX(self, heatmap):
        '''
        input parameters:
            heatmap: 2D Tensor ((1), H, W)
        output parameters:
            heatmap_nms: 2D numpy array (H, W), key point will be labeled as +1
            keypts: [3, n] n: the number of key points, 1st row: x; 2nd row: y, 3rd row: score
            keypts_inds: [n,] indices of key points
            Note: keypts and keypts_inds are numpy arrays.
        '''
        # 篩選出 heatmap score數值高於 conf_thresh 的點
        heatmap   = heatmap.squeeze()
        H, W      = heatmap.shape[0], heatmap.shape[1]        # H=120; W=160
        ys, xs    = torch.where(heatmap >= self.conf_thresh)  # ys:(9549,), xs:(9549,)
        pts       = torch.zeros((3, len(xs)))                 # pts:(3,9549)
        pts[0, :] = xs
        pts[1, :] = ys
        pts[2, :] = heatmap[ys, xs]
                
        # 根據 score 進行降冪排序(由大至小)
        inds1     = torch.argsort(-pts[2,:]) # inds1: (9549,)
        corners   = pts[:, inds1]            # corners:[xs, ys, score] (3,9549)
        rcorners  = corners[:2, :].int()     # rcorners:[xs, ys] (2,9549)
                
        # 測試 rcorners 是否為0點或僅為1點
        if rcorners.shape[1] == 0:
            return torch.zeros((3,0), dtype=torch.int32), torch.zeros(0, dtype=torch.int32)
        if rcorners.shape[1] == 1:
            out = torch.vstack((rcorners, pts[2])).reshape(3,1)
            return out, torch.zeros(1, dtype=torch.int32)
        
        # 記錄特徵點格點位置及其索引 
        grid = torch.zeros((H,W), dtype=torch.int32)      # grid:(120,160)
        inds = torch.zeros((H,W), dtype=torch.int32)      # inds:(120,160)
        for i, rc in enumerate(rcorners.T):
            grid[rc[1], rc[0]] = 1                 
            inds[rc[1], rc[0]] = i                 
        
        # Iterate through points, highest to lowest confidence, suppress neighborhood
        # Note: This method ignores the key points in the border, i.e., 0<=X<pad,
        # W-pad<=X<W, 0<=Y<pad and H-pad<=Y<H, and therefore, it does'nt affect 
        # the key points inside the border.
        # 註：若特徵點位於邊界內，此特徵點將不予考慮，以免影響到邊界內的特徵點
        pad   = self.nms_dist                       # 9x9 patches: nms
        count = 0
        for i, rc in enumerate(rcorners.T):
            pt = (rc[0], rc[1])                     # rc[0]:xs, rc[1]:ys, grid[y,x]
            if grid[pt[1], pt[0]] == 1:
                if (pt[0]-pad)>=0 and (pt[0]+pad)<W and (pt[1]-pad)>=0 and (pt[1]+pad)<H:
                    grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                    grid[pt[1], pt[0]] = -1
                    count += 1
                else:
                    grid[pt[1], pt[0]] = 0          # count = 423 
        
        # Get all surviving -1's and return sorted array of remaining corners
        keepy, keepx = torch.where(grid == -1)         # keepy:(423,), keepx:(423,)
        inds_keep    = inds[keepy, keepx]              # inds_keep:(423,)
        keypts       = corners[:, inds_keep.numpy()]   # corners:(3,9549); keypts:(3,423)
        inds2        = torch.argsort(-keypts[2,:])     # inds2:(423,)
        keypts       = keypts[:, inds2]                # keypts:(3,423)
        keypts_inds  = inds1[inds_keep[inds2].numpy()] # keypts_inds:(423,)
                                                       # inds1是score由大排至小的index 
        heatmap_nms  = torch.zeros_like(heatmap)       # heatmap_nms:(120,160)
        heatmap_nms[keepy, keepx] = 1
        
        # To show output variables as numpy nd-array
        heatmap_nms_np = heatmap_nms.detach().cpu().clone().numpy()
        keypts_np      = keypts.detach().cpu().clone().numpy()
        keypts_inds_np = keypts_inds.detach().cpu().clone().numpy()
                
        return heatmap_nms, keypts, keypts_inds
        # 輸出: heatmap_nms:(120,160), keypts:(3,423) and keypts_inds:(423,)

    def pred_soft_argmax(self, labels_2D, heatmap):   # labels_2D:[2,1,120,160]
        '''                                           # heatmap:[2,1,120,160]
            input parameters:
                labels_2D: heatmap_nms_batch[bs, 1, H, W]. 
                heatmap: [bs, 1, H, w]. H and W are image height and width.
            output parameters:
                outs: {'pred': dxdy, 'patches': patches}
                dxdy: centroid offset of patches; patches: roi-pooling region
        '''
        outs       = {}    
        patch_size = self.patch_size                         # patch_size = 5
        label_idx  = labels_2D[...].nonzero()                # label_idx:[423,4]
                                                             # label_idx:[423,(bs,c,y,x)]
        # pts_to_bbox 
        points  = label_idx[:, 2:]                           
        shift_L = (patch_size + 1)/2                         # shift_L = 3
        shift_R = patch_size - shift_L                       # shift_R = 3
        pts_L   = points - shift_L
        pts_R   = points + shift_R + 1                       # bbox:[423,4]->(x1,y1,x2,y2)
        bbox    = torch.stack((pts_L[:,1], pts_L[:,0], pts_R[:,1], pts_R[:,0]), dim=1)
        
        rois    = torch.cat((label_idx[:, :1], bbox), dim=1) # rois:[423,5]->(bs,x1,y1,x2,y2)
        # roi-pooling: rois(7x7) -> patches(5x5) 自7x7區塊中，利用roi-pooling抽取5x5區塊
        patches = roi_pool(heatmap, rois.float(), (patch_size, patch_size), spatial_scale=1.0)
                                                             # patches:[423,1,5,5]
        # do log-transform for patches
        patches[patches<0] = 1e-6
        patches_log = torch.log(patches)                     # patches_log:[423,1,5,5] 
        
        # roi-pooling and calculating centroid of patches(計算5x5區塊的質心)
        m    = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=False)
        dxdy = m(patches_log)                                # dxdy:[423,1,2] (centroid of patches)
        
        # extract offsets(計算5x5區塊質心偏移量)
        dxdy = dxdy.squeeze(dim=1)                           # dxdy:[423,2]->[423,(x,y)]
        dxdy = dxdy-patch_size//2                            # dxdy:[423,2]; patch_size=5
        
        # loss
        outs['pred']    = dxdy                               # outs['pred']:[423,2]->[423,(x,y)]
        outs['patches'] = patches                            # outs['patches']:[423,1,5,5]
        return outs
        
    def crop_or_pad_choice(self, in_num_points, out_num_points, shuffle=False):
        """Crop or pad point cloud to a fixed number; return the indexes
        Args:
            points (np.ndarray): point cloud. (n, d)
            num_points (int): the number of output points
            shuffle (bool): whether to shuffle the order
        Returns:
            np.ndarray: output point cloud
            np.ndarray: index to choose input points
        """
        if shuffle:
            choice = np.random.permutation(in_num_points)
        else:
            choice = np.arange(in_num_points)
        
        assert out_num_points > 0, 'out_num_points = %d must be positive int!'%out_num_points
        
        if in_num_points >= out_num_points:
            choice = choice[:out_num_points]
        else:
            num_pad = out_num_points - in_num_points
            pad     = np.random.choice(choice, num_pad, replace=True)
            choice  = np.concatenate([choice, pad])
        
        return choice

    def sample_desc_from_points(self, coarse_desc, pts, cell_size=8):
        """
         inputs:
            coarse_desc: tensor [1,256,Hc,Wc]
            pts: tensor [N,2]->[N,(x,y)] (should be the same device as desc)
         return:
            desc: tensor [1,N,D]
        """
        # --- Process descriptor.
        samp_pts = pts.transpose(0,1)                                # samp_pts:[2(x,y),423]
        H, W     = coarse_desc.shape[2]*cell_size, coarse_desc.shape[3]*cell_size
        D        = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            # desc = torch.zeros((D, 0))
            desc = torch.ones((1, 1, D))
        else:
            # Interpolate into descriptor map using 2D point locations.
            # samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts       = samp_pts.transpose(0, 1).contiguous()   # samp_pts:[423,2(x,y)]
            samp_pts       = samp_pts.view(1, 1, -1, 2)              # samp_pts:[1,1,423,2] 
            samp_pts       = samp_pts.float()
            samp_pts       = samp_pts.to(self.device)
            
            # desc:[1,256,1,423]; coarse_desc:[1,256,15,20]
            desc = nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True) 
            # desc = desc.data.cpu().numpy().reshape(D, -1)
            # desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]    # desc.squeeze():[256,423]
            desc = desc.squeeze().transpose(0,1).unsqueeze(0)        # desc:[1,423,256]
        
        return desc
    
    def batch_extract_features(self, desc, heatmap_nms_batch, residual):
        """
        input parameters:
          desc:[2,256,15,20] output from descriptor decoder
          heatmap_nms_batch:[2,1,120,160] output from interest point decoder
          residual:[831,2]->[831,(x,y)] 5x5區塊質心偏移量
        output parameters: -- type: tensorFloat            
          pts: tensor [batch, N, 2] (no grad)  (x, y)       
          pts_offset: tensor [batch, N, 2] (grad) (x, y)    
          pts_desc: tensor [batch, N, 256] (grad)
        """
        batch_size = heatmap_nms_batch.shape[0]             # batch_size = 2
        
        pts_int, pts_offset, pts_desc = [], [], []
        pts_idx = heatmap_nms_batch[...].nonzero()          # [831,4(batch,c,y,x)]
        
        for i in range(batch_size):
            mask_b    = (pts_idx[:,0] == i)                 # mask_b:[831,]
            pts_int_b = pts_idx[mask_b][:,2:].float()       # pts_int_b:[423,2(y,x)]
            pts_int_b = pts_int_b[:,[1, 0]]                 # res_int_b:[423,2(x,y)]
            res_b     = residual[mask_b]                    # res_b:[423,2(x,y)]
            pts_b     = pts_int_b + res_b                   # pts_b:[423,2(x,y)]
            
            # extract desc                                  # pts_desc_b:[423,256]
            pts_desc_b = self.sample_desc_from_points(desc[i].unsqueeze(0), pts_b).squeeze(0)
                        
            # get random shuffle
            choice = self.crop_or_pad_choice(pts_int_b.shape[0], 
                                             out_num_points=self.out_num_points, 
                                             shuffle=True)
            # choice:[500,] Array of int32
            # choice = torch.tensor(choice)                   # choice:[500,]
            pts_int.append(pts_int_b[choice])                 # 記錄特徵點(x,y)座標
            pts_offset.append(res_b[choice])                  # 記錄5x5區塊質心偏移量
            pts_desc.append(pts_desc_b[choice])               # 記錄特徵點256維的描述子

        pts_int    = torch.stack((pts_int), dim=0)            # pts_int:[2,500,2]
        pts_offset = torch.stack((pts_offset), dim=0)         # pts_offset:[2,500,2]
        pts_desc   = torch.stack((pts_desc), dim=0)           # pts_desc:[2,500,256]
        return {'pts_int': pts_int, 'pts_offset': pts_offset, 'pts_desc': pts_desc}
        
    def process_outputX(self): 
        """
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        # (1) depth to space: semi[2,65,15,20] -> heatmap[2,1,120,160]
        output   = self.output
        blk_size = 8
        semi     = output['semi']                       # semi:[2,65,15,20]
        desc     = output['desc']                       # desc:[2,256,15,20]
        
        # channel-wise softmax for semi
        dense  = nn.functional.softmax(semi, dim=1)     # dense:[2,65,15,20]
        
        # Remove dustbin
        nodust = dense[:, :-1, :, :]                    # nodust:[2,64,15,20]
        
        # depth to space conversion
        bs, d_depth, d_height, d_width = nodust.size()
        heatmap = nodust.permute(0, 2, 3, 1)                                 # heatmap:[2,15,20,64]
        heatmap = heatmap.reshape(bs, d_height, d_width, blk_size, blk_size) # heatmap:[2,15,20,8,8]
        heatmap = heatmap.permute(0, 1, 3, 2, 4)                             # heatmap:[2,15,8,20,8]
        heatmap = heatmap.reshape(bs, d_height*blk_size, d_width*blk_size)   # heatmap:[2,120,160]
        heatmap = heatmap.unsqueeze(dim=1)                                   # heatmap:[2,1,120,160]
        
        # (2) heatmap to heatmap_nms_batch: [2,120,160] -> [2,120,160]
        heatmap_nms_L = []
        keypts_L      = []
        keypts_inds_L = []
        for h in heatmap:
            heatmap_nms, keypts, keypts_inds = self.heatmap_nmsX(h)
            heatmap_nms_L.append(heatmap_nms)
            keypts_L.append(keypts)
            keypts_inds_L.append(keypts_inds)
        
        # heatmap_nms_batch:[2,1,120,160]
        for i in range(len(heatmap_nms_L)):
            heatmap_nms_L[i] = heatmap_nms_L[i].unsqueeze(dim=0)
        heatmap_nms_batch = torch.vstack(heatmap_nms_L).unsqueeze(dim=1)
                
        # (3) 計算5x5區塊的質心偏移量
        # outs={'pred':dxdy[423,2]->[423,(x,y)], 'patches':patches[423,1,5,5]}
        outs     = self.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']                   # residual:[423,2]->[423,(x,y)]
                
        # (4) 計算特徵點256維的描述子                  
        outs = self.batch_extract_features(desc, heatmap_nms_batch, residual)
        # outs['pts_int']:[2,500,2]    記錄特徵點(x,y)座標
        # outs['pts_offset']:[2,500,2] 記錄5x5區塊質心偏移量(roi-pooling:7x7->5x5)
        # outs['pts_desc']:[2,500,256] 記錄特徵點256維的描述子
        
        # output.update({'semi':semi, 'desc':desc, 
        #    'pts_int':pts_int, 'pts_offset':pts_offset, 'pts_desc':pts_desc})
        output.update(outs)
        self.output = output
        return output

def print_dict_attr(dictionary, attr=None, file=None):
    for item in list(dictionary): # list(dict): return only the keys of dictionary
        d = dictionary[item]      # d: Tensor has 'shape' attribute
        if attr == None:
            print(item, ": ", d, file=file)
        else:
            if hasattr(d, attr):
                print(item, ": ", getattr(d, attr), file=file)
            else:
                print(item, ": ", len(d), file=file)

def get_matches(SP_desc):                # SP_desc:[2]=>[500,256]
    from model_wrap import PointTracker
    tracker = PointTracker(max_length=2, nn_thresh=1.3)
    # print("SP_desc[1]: ", SP_desc[1].shape)
    
    f = lambda x: x.cpu().detach().numpy()
    matching_mask = tracker.nn_match_two_way(f(SP_desc[0]).T, f(SP_desc[1]).T, nn_thresh=1.3)
    return matching_mask

def main(): 
    # Instantiation of superpoint model
    model = SuperPointNetX()
    print("[INFO] This device is using {}".format(model.device))
    model = model.to(model.device)             # put model into the device
    # summary(model, input_size=(1, 120, 160))
    summary(model, input_size=(1, 240, 320))
        
    # detect keypoints and output its descriptor    
    # image      = torch.rand((2,1,120,160))
    image      = torch.rand((2,1,240,320))
    model_outs = model(image.to(model.device))       # put image data into the device
    print("\nmodel_outs: ", list(model_outs))
    print_dict_attr(model_outs, attr='shape')
    print()
    
    # 程式測試 
    import time
    from tqdm import tqdm
    iter_max = 50
    
    # 特徵點偵測與描述子處理    
    start = time.time()
    print("Start timer!")
    SP_pts, SP_desc, SP_offset = [], [], []
    for i in tqdm(range(iter_max)):
        outs = model(image.to(model.device))
        SP_pts.append(outs['pts_int'].squeeze())       # SP_pts:[50]=>[2,500,2]
        SP_desc.append(outs['pts_desc'].squeeze())     # SP_desc:[50]=>[2,500,256]
        SP_offset.append(outs['pts_offset'].squeeze()) # SP_offset:[50]=>[2,500,2]
    end = time.time()
    print("forward + process output: ", iter_max/(end - start), " iter/s")
    
    # 特徵點匹配處理
    start = time.time()
    print("Start timer!")                              # len(SP_pts) = 50
    for i in tqdm(range(len(SP_pts))):                 # SP_desc[0].shape:[2,500,256]
        get_matches([SP_desc[i][0], SP_desc[i][1]])    # SP_desc[i][0]:[500,256]
    end = time.time()                                  # SP_desc[i][1]:[500,256]
    print()
    print("nn matches: ", iter_max/(end - start), " iters/s")

if __name__ == '__main__':
    main()



