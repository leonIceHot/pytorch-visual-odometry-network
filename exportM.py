"""
This script exports detection/ description using pretrained model.

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

## basic
import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
from imageio import imread
from tqdm import tqdm
from tensorboardX import SummaryWriter

## torch
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

## other functions
from utils.utils import (
    tensor2array,
    save_checkpoint,
    load_checkpoint,
    save_path_formatter,
)
from utils.utils import getWriterPath
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.utils import inv_warp_image_batch
from models.model_wrap import SuperPointFrontend_torch, PointTracker

## parameters
from settings import EXPER_PATH

#### util functions


def combine_heatmap(heatmap, inv_homographies, mask_2D, device="cpu"):
    ## multiply heatmap with mask_2D
    heatmap = heatmap * mask_2D # [100,1,240,320]

    heatmap = inv_warp_image_batch( # [100,1,240,320]
        heatmap, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )

    ##### check
    mask_2D = inv_warp_image_batch( # [100,1,240,320]
        mask_2D, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )
    heatmap = torch.sum(heatmap, dim=0) # [1,240,320]
    mask_2D = torch.sum(mask_2D, dim=0) # [1,240,320]
    return heatmap / mask_2D
    pass


#### end util functions


def export_descriptor(config, output_dir, args):
    """
    # input 2 images, output keypoints and correspondence
    save prediction:
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)
            'matches': np [N3, 4]
    """
    from utils.loader import get_save_path
    from utils.var_dim import squeezeToNumpy

    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train on device: cuda:0
    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(getWriterPath(task=args.command, date=False))
    save_path = get_save_path(output_dir)
    # will save everything to logs\superpoint_hpatches_testI\checkpoints
    save_output = save_path / "../predictions"
    os.makedirs(save_output, exist_ok=True)

    ## parameters
    outputMatches = True
    subpixel = config["model"]["subpixel"]["enable"]              # True
    patch_size = config["model"]["subpixel"]["patch_size"]        # 5

    # data loading
    from utils.loader import dataLoader_test as dataLoader
    task = config["data"]["dataset"]                              # hpatches
    #  workers_test: 1
    data = dataLoader(config, dataset=task)                       
    test_set, test_loader = data["test_set"], data["test_loader"] # 580
    from utils.print_tool import datasize
    # test split size 580 in 580 batches
    datasize(test_loader, config, tag="test")

    # model loading
    from utils.loader import get_module
    Val_model_heatmap = get_module("", config["front_end_model"])
    
    ## load pretrained
    val_agent = Val_model_heatmap(config["model"], device=device)
    # creating model: SuperPointNet_gauss2
    # successfully load pretrained model from: logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar
    val_agent.loadModel()

    ## tracker
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    ###### check!!!
    count = 0
    for i, sample in tqdm(enumerate(test_loader)): # 580
        img_0, img_1 = sample["image"], sample["warped_image"] #[1,1,240,320]


        # img = img_0
        def get_pts_desc_from_agent(val_agent, img, device="cpu"):
            """
            pts: list [numpy (3, N)]
            desc: list [numpy (256, N)]
            """
            heatmap_batch = val_agent.run(
                img.to(device)
            )  # heatmap: numpy [batch, 1, H, W]
            # heatmap to pts
            pts = val_agent.heatmap_to_pts()
            # print("pts: ", pts)
            if subpixel:
                pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
            # heatmap, pts to desc
            desc_sparse = val_agent.desc_to_sparseDesc()
            # print("pts[0]: ", pts[0].shape, ", desc_sparse[0]: ", desc_sparse[0].shape)
            # print("pts[0]: ", pts[0].shape)
            outs = {"pts": pts[0], "desc": desc_sparse[0]}
            return outs

        def transpose_np_dict(outs):
            for entry in list(outs):
                outs[entry] = outs[entry].transpose()

        # first image, no matches
        # heatmap:  (244, 324)
        outs = get_pts_desc_from_agent(val_agent, img_0, device=device)

        # pts: np [3, N] (3, 520) 
        # desc: (256, 520)
        pts, desc = outs["pts"], outs["desc"]  

        if outputMatches == True:
            tracker.update(pts, desc)

        # save keypoints
        pred = {"image": squeezeToNumpy(img_0)}
        pred.update({"prob": pts.transpose(), "desc": desc.transpose()})

        # second image, output matches
        outs = get_pts_desc_from_agent(val_agent, img_1, device=device)
        pts, desc = outs["pts"], outs["desc"]

        if outputMatches == True:
            tracker.update(pts, desc)

        # from utils.var_dim import squeezeToNumpy
        pred.update({"warped_image": squeezeToNumpy(img_1)})
        # print("total points: ", pts.shape)
        pred.update(
            {
                "warped_prob": pts.transpose(),
                "warped_desc": desc.transpose(),
                "homography": squeezeToNumpy(sample["homography"]),
            }
        )

        if outputMatches == True:
            matches = tracker.get_matches()
            print("matches: ", matches.transpose().shape) # matches:  (452, 4) 
            pred.update({"matches": matches.transpose()})
        print("pts: ", pts.shape, ", desc: ", desc.shape) 
        # pts:  (3, 625) , desc:  (256, 625)

        # clean last descriptor
        tracker.clear_desc()
        
        # Save path
        filename = str(count)
        # logs/superpoint_hpatches_testI/checkpoints/../predictions/0.npz'
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)
        print("save: ", path)
        count += 1
    print("output pairs: ", count)


@torch.no_grad()
def export_detector_homoAdapt_gpu(config, output_dir, args):
    """
    input 1 images, output pseudo ground truth by homography adaptation.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    from utils.utils import pltImshow
    from utils.utils import saveImg
    from utils.draw import draw_keypoints

    # basic setting
    task = config["data"]["dataset"] # Coco
    export_task = config["data"]["export_folder"] # train val 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(
        getWriterPath(task=args.command, exper_name=args.exper_name, date=False) # 2021/09/25
    )

    ## parameters
    nms_dist = config["model"]["nms"]                               # 4
    top_k = config["model"]["top_k"]                                # 600
    homoAdapt_iter = config["data"]["homography_adaptation"]["num"] # 100
    conf_thresh = config["model"]["detection_threshold"]            # 0.015
    nn_thresh = 0.7 
    outputMatches = True
    count = 0
    max_length = 5
    output_images = args.outputImg
    check_exist = True

    ## save data
    save_path = Path(output_dir)  # logs/magicpoint_synth_homoAdapt_cocoI                     
    # logs/magicpoint_synth_homoAdapt_cocoI
    save_output = save_path       
    # logs/magicpoint_synth_homoAdapt_cocoI/predictions/val
    save_output = save_output / "predictions" / export_task
    # logs/magicpoint_synth_homoAdapt_cocoI/checkpoints
    save_path = save_path / "checkpoints"
    # will save everything to logs\magicpoint_synth_homoAdapt_cocoI\checkpoints
    logging.info("=> will save everything to {}".format(save_path))
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_output, exist_ok=True)

    # data loading
    from utils.loader import dataLoader_test as dataLoader

    data = dataLoader(config, dataset=task, export_task=export_task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    # model loading
    ## load pretrained
    try:
        # model:  SuperPointNet_gauss2
        path = config["pretrained"]
        print("==> Loading pre-trained network.")
        # logs\magicpoint_synth\checkpoints\superPointNet_200000_checkpoint.pth.tar
        print("path: ", path)
        # This class runs the SuperPoint network and processes its outputs.

        fe = SuperPointFrontend_torch(
            config=config,
            weights_path=path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=False,
            device=device,
        )
        print("==> Successfully loaded pre-trained network.")

        fe.net_parallel() # === Let's use 1 GPUs!
        print(path) # logs/magicpoint_synth/checkpoints/superPointNet_200000_checkpoint.pth.tar
        
        # save to files
        # logs/magicpoint_synth_homoAdapt_cocoI/predictions/val/export.txt
        save_file = save_output / "export.txt"
        with open(save_file, "a") as myfile:            
            myfile.write("load model: " + path + "\n")
    except Exception:
        print(f"load model: {path} failed! ")
        raise

    def load_as_float(path):
        return imread(path).astype(np.float32) / 255

    tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)
    with open(save_file, "a") as myfile:
        myfile.write("homography adaptation: " + str(homoAdapt_iter) + "\n")

    ## loop through all images
    # test_loader 40504
    for i, sample in tqdm(enumerate(test_loader)):         
        img, mask_2D = sample["image"], sample["valid_mask"] # img [1,100,240,320] mask_2D [1,100,240,320]       
        img = img.transpose(0, 1) # img [100,1,240,320] 
        # sample[240,320]                
        img_2D = sample["image_2D"].numpy().squeeze()  # img_2D   [240,320]
        mask_2D = mask_2D.transpose(0, 1)              # mask_2D  [100,1,240,320]

        inv_homographies, homographies = (
            sample["homographies"],                    # [1,100,3,3]
            sample["inv_homographies"],                # [1,100,3,3]
        )
        img, mask_2D, homographies, inv_homographies = (
            img.to(device),                            # [100,1,240,320]
            mask_2D.to(device),                        # [100,1,240,320]
            homographies.to(device),                   # [1,100,3,3]
            inv_homographies.to(device),               # [1,100,3,3]
        )
        # sample = test_set[i]
        name = sample["name"][0]      # COCO_train2014_000000000009
        logging.info(f"name: {name}") # [03/20/2022 09:24:33 INFO] name: COCO_train2014_000000000009
        if check_exist:
            p = Path(save_output, "{}.npz".format(name)) # logs/magicpoint_synth_homoAdapt_cocoiii/predictions/train/COCO_train2014_000000000009.npz
            if p.exists():
                logging.info("file %s exists. skip the sample.", name) # [03/20/2022 09:26:13 INFO] file COCO_train2014_000000000009 exists. skip the sample.
                continue

        # pass through network
        heatmap = fe.run(img, onlyHeatmap=True, train=False) # [100,1,240,320]
        outputs = combine_heatmap(heatmap, inv_homographies, mask_2D, device=device) # [1,240,320]
        pts = fe.getPtsFromHeatmap(outputs.detach().cpu().squeeze())  # (x,y, prob) (3,53)

        # subpixel prediction
        if config["model"]["subpixel"]["enable"]:
            fe.heatmap = outputs  # tensor [batch, 1, H, W]
            print("outputs: ", outputs.shape) #  outputs:  torch.Size([1, 240, 320])
            print("pts: ", pts.shape)         #  pts:  (3, 53)
            pts = fe.soft_argmax_points([pts])#  heatmap:  (244, 324)
            pts = pts[0]                      #  (3,53)

        ## top K points
        pts = pts.transpose()
        print("total points: ", pts.shape)    # total points:  (53, 3)
        '''
            pts:  [[2.97215071e+02 7.34162602e+01 2.07455352e-01]
                   [2.93676806e+02 1.65935910e+02 1.92684427e-01]
                   [2.27044928e+02 1.99157301e+02 1.49741307e-01]
                   [1.33776341e+02 2.10454314e+02 1.45094782e-01]
                   [2.54426469e+02 1.97883073e+02 1.25162661e-01]]        
        '''
        print("pts: ", pts[:5])
        if top_k:
            if pts.shape[0] > top_k:
                pts = pts[:top_k, :]
                print("topK filter: ", pts.shape)

        ## save keypoints
        pred = {}
        pred.update({"pts": pts})

        ## - make directories
        filename = str(name)
        if task == "Kitti" or "Kitti_inh":
            scene_name = sample["scene_name"][0]
            os.makedirs(Path(save_output, scene_name), exist_ok=True)

        # logs/magicpoint_synth_homoAdapt_cocoI/predictions/val/COCO_val2014_000000000544.npz
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)

        ## output images for visualization labels
        if output_images:
            img_pts = draw_keypoints(img_2D * 255, pts.transpose()) # (720,960,3)
            # 'logs/magicpoint_synth_homoAdapt_cocoiii/predictions/train/0.png'
            f = save_output / (str(count) + ".png") 
            if task == "Coco" or "Kitti":
                f = save_output / (name + ".png")
            saveImg(img_pts, str(f))
        count += 1

    print("output pseudo ground truth: ", count)
    save_file = save_output / "export.txt"
    with open(save_file, "a") as myfile:
        myfile.write("Homography adaptation: " + str(homoAdapt_iter) + "\n")
        myfile.write("output pairs: " + str(count) + "\n")
    pass


if __name__ == "__main__":
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # 2: export_detector_homoAdapt ==> Step II
    # 4: export_descriptor         ==> Step IV 
    train_target = 2              
    
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    
    # using homography adaptation to export detection psuedo ground truth
    if train_target==2: # Step II Exporting detections on MS-COCO
        # export command: export_detector_homoAdapt
        ap.add_argument("--command", type=str, default='export_detector_homoAdapt')               
        # export coco - do on training set 
        # export coco - do on validation set
        ap.add_argument("--config", type=str,default='configs/magicpoint_coco_export.yaml')      
        ap.add_argument("--exper_name", type=str,default='magicpoint_synth_homoAdapt_cocoX')
        ap.add_argument("--eval", action="store_true")
        ap.add_argument("--outputImg", action="store_true",default=True, help="output image for visualization")
        ap.add_argument("--debug", action="store_true", default=True, help="turn on debuging mode")               

    elif train_target==4:  # Step IV Export the metrics on HPatches        
        # export command: export_descriptor     
        ap.add_argument("--command", type=str, default='export_descriptor')        
        ap.add_argument("--config", type=str,default='configs/magicpoint_repeatability_heatmap.yaml')       
        ap.add_argument("--exper_name", type=str,default='superpoint_hpatches_test_vonetIII')
        ap.add_argument("--correspondence", action="store_true")
        ap.add_argument("--eval", action="store_true",default=True)
        ap.add_argument("--debug", action="store_true", default=True, help="turn on debuging mode")
    else:
        print('Training mode must be 2 , 4. Others are not accepted!')
        exit(1)    

    # parse commands
    args = ap.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    print("check config!! ", config)

    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info("Running command {}".format(args.command.upper()))
    
    # args.func(config, output_dir, args)    
    if args.command=='export_descriptor':
        export_descriptor(config, output_dir, args)
    elif args.command=='export_detector_homoAdapt':
        export_detector_homoAdapt_gpu(config, output_dir, args)
    else:
        exit(1)
    
