"""Training script
This is the training script for superpoint detector and descriptor.

python train4.py train_base --eval configs/magicpoint_shapes_pair.yaml magicpoint_synth

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import argparse
import yaml
import os
import logging

import torch
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter

# from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
from utils.utils import getWriterPath
from settings import EXPER_PATH

## loaders: data, model, pretrained model
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.logging import *
# from models.model_wrap import SuperPointFrontend_torch, PointTracker

###### util functions ######
def datasize(train_loader, config, tag='train'):
    if tag == 'train':
        logging.info('== %s split size %d in %d batches'%\
        (tag, len(train_loader)*config['model']['batch_size'], len(train_loader)))
    elif tag == 'val':
        logging.info('== %s split size %d in %d batches'%\
        (tag, len(train_loader)*config['model']['eval_batch_size'], len(train_loader)))  
    else:            
        pass
    
from utils.loader import get_save_path
###### util functions end ######


###### train script ######
def train_base(config, output_dir, args):
    return train_joint(config, output_dir, args)
    pass


def train_joint(config, output_dir, args):
    assert 'train_iter' in config

    # config
    # from utils.utils import pltImshow
    # from utils.utils import saveImg
    torch.set_default_tensor_type(torch.FloatTensor)
    # dataset: Coco
    task = config['data']['dataset']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    # writer = SummaryWriter(getWriterPath(task=args.command, date=True))   # 2021/09/23
    writer = SummaryWriter(getWriterPath(task=args.command,                 # 2021/09/23
                          exper_name=args.exper_name, date=False))
    ## save data
    # will save everything to logs\superpoint_cocoIII\checkpoints
    save_path = get_save_path(output_dir)

    # data loading
    # data = dataLoader(config, dataset='syn', warp_input=True)

    # load labels from:  logs\magicpoint_synth_homoAdapt_cocoII\predictions/train
    # load labels from:  logs\magicpoint_synth_homoAdapt_cocoII\predictions/val
    data = dataLoader(config, dataset=task, warp_input=True)
    # train_loader 10348 
    # val_loader 5063
    train_loader, val_loader = data['train_loader'], data['val_loader']
    
    # train split size 82784 in 10348 batches
    datasize(train_loader, config, tag='train')
    
    # val split size 40504 in 5063 batches
    datasize(val_loader, config, tag='val')
    
    # init the training agent using config file
    
    # from train_model_frontend import Train_model_frontend
    from utils.loader import get_module
    train_model_frontend = get_module('', config['front_end_model'])

    train_agent = train_model_frontend(config, save_path=save_path, device=device)

    # writer from tensorboard
    train_agent.writer = writer

    # feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    # load model initiates the model and load the pretrained model (if any)
    train_agent.loadModel()
    # adam optimizer
    train_agent.dataParallel()

    try:
        # train function takes care of training and evaluation
        train_agent.train()
    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")
        train_agent.saveModel()
        pass

if __name__ == '__main__':
    # global var
    # 1: train_base mode  ==> Training MagicPoint on Synthetic Shapes
    # 3: train_joint mode ==> Training Superpoint on MS-COCO
    train_target =  3               
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    if train_target==1: # Step I ==> Training MagicPoint on Synthetic Shapes   
        # training command: train_base 
        ap.add_argument("--command", type=str, default='train_base')
        ap.add_argument("--config", type=str, default='configs/magicpoint_shapes_pair.yaml')
        ap.add_argument("--exper_name", type=str, default='magicpoint_synth_vonetV')
        ap.add_argument("--eval", action='store_true', default=True)
        ap.add_argument("--debug", action='store_true', default=True, help='turn on debugging mode')
    elif train_target==3: # Step III ==> ==> Training Superpoint on MS-COCO
        # training command: train_joint 
        ap.add_argument("--command", type=str, default='train_joint')
        ap.add_argument("--config", type=str, default='configs/superpoint_coco_train_heatmap.yaml')
        ap.add_argument("--exper_name", type=str, default='superpoint_coco_vonetX')
        ap.add_argument("--eval", action='store_true', default=True)
        ap.add_argument("--debug", action='store_true', default=True, help='turn on debugging mode')        
    else:
        print('Training mode must be 1 or 3. Others are not accepted!')
        exit(1)
        
    # parse commands
    args = ap.parse_args()
    if args.debug:
       logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                       datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    with open(args.config, 'r') as f:
        # config = yaml.load(f)
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # EXPER_PATH from settings.py
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    print(f'output directory is {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info('Running command {}'.format(args.command.upper()))
    if args.command=='train_base':
        train_base(config, output_dir, args)
    elif args.command=='train_joint':
        train_joint(config, output_dir, args)
    else:
        exit(1)


