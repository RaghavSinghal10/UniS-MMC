import time
import random
import torch
torch.cuda.current_device()
import logging
import argparse
import os
import numpy as np
import warnings

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from model.model import MMC
from src.train_food101_2 import train
from src.config import Config
from src.functions import dict_to_str

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):

    args.name_seed = args.name + '_' + str(args.seed)
    
    setup_seed(args.seed)

    if args.dataset in ['Food101', 'N24News']:
        results = train(args)

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MMC',
                        help='project name')
    parser.add_argument('--dataset', type=str, default='Food101',
                        help='support N24News/Food101')
    parser.add_argument('--text_type', type=str, default='headline',
                        help='support headline/caption/abstract')
    parser.add_argument('--mmc', type=str, default='UniSMMC',
                        help='support UniSMMC/UnSupMMC/SupMMC')
    parser.add_argument('--mmc_tao', type=float, default=0.07,
                        help='use supervised contrastive loss or not')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--lr_mm', type=float, default=1e-3,
                        help='--lr_mm')
    parser.add_argument('--min_epoch', type=int, default=1,
                        help='min_epoch')    
    parser.add_argument('--valid_step', type=int, default=50,
                        help='valid_step')              
    parser.add_argument('--max_length', type=int, default=512,
                        help='max_length')
    parser.add_argument('--text_encoder', type=str, default='bert_base',
                        help='bert_base/roberta_base/bert_large')
    parser.add_argument('--image_encoder', type=str, default='vit_base',
                        help='vit_base/vit_large')
    parser.add_argument('--text_out', type=int, default=768,
                        help='text_out')
    parser.add_argument('--img_out', type=int, default=768,
                        help='img_out')                                        
    parser.add_argument('--lr_mm_cls', type=float, default=1e-3,
                        help='--lr_mm_cls')
    parser.add_argument('--mm_dropout', type=float, default=0.0,
                        help='--mm_dropout')
    parser.add_argument('--lr_text_tfm', type=float, default=2e-5,
                        help='--lr_text_tfm')
    parser.add_argument('--lr_img_tfm', type=float, default=5e-5,
                        help='--lr_img_tfm')
    parser.add_argument('--lr_img_cls', type=float, default=1e-4,
                        help='--lr_img_cls')
    parser.add_argument('--lr_text_cls', type=float, default=5e-5,
                        help='--lr_text_cls')
    parser.add_argument('--text_dropout', type=float, default=0.0,
                        help='--text_dropout')
    parser.add_argument('--img_dropout', type=float, default=0.1,
                        help='--img_dropout')
    parser.add_argument('--nplot', type=str, default='',
                        help='MTAV')
    parser.add_argument('--data_dir', type=str, default='./datasets/',
                        help='support wmsa') 
    parser.add_argument('--test_only', type=bool, default=False,
                        help='train+test or test only')
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained_models',
                        help='path to pretrained models from Hugging Face.')
    parser.add_argument('--model_save_dir', type=str, default='Path/To/results/models',
                        help='path to save model parameters.')
    parser.add_argument('--res_save_dir', type=str, default='Path/To/results/results',
                        help='path to save training results.')
    parser.add_argument('--fig_save_dir', type=str, default='Path/To/results/imgs',
                        help='path to save figures.')
    parser.add_argument('--logs_dir', type=str, default='Path/To/results/logs',
                        help='path to log results.')  # NO
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--seeds', default=1, nargs='+', type=int, help='set seeds for multiple runs!')
    return parser.parse_args()

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")

    args = parse_args()
    config = Config(args)
    args = config.get_config()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    args.device = device
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    run(args)


