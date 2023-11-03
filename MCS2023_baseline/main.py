import argparse
import os
import os.path as osp
import random
import sys
import yaml

import torch
import numpy as np

import utils as utilities

from tqdm import tqdm

from data import get_dataloader
from models import Model
from train import train, val
from get_dataloader import get_dataloader

import glob
import time
import random
import math

# DATALOADER
import cv2
from PIL import Image
import numpy as np
import albumentations as A
import torchvision.transforms as T
from PIL import Image
import pandas as pd

# BUILDING MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F

# TRAINING
from torch.utils.data import DataLoader, Dataset
import faiss
from tqdm import tqdm_notebook as tqdm

# OTHER STUFF
import timm
from transformers import (get_linear_schedule_with_warmup, 
                          get_constant_schedule,
                          get_cosine_schedule_with_warmup, 
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_constant_schedule_with_warmup)
import gc
import transformers
from transformers import CLIPProcessor, CLIPVisionModel,  CLIPVisionConfig
# from pytorch_metric_learning import losses
import open_clip
from dataset import aicrowd_data_loader


class CFG:
    model_name = 'ViT-L-14-336'
    model_data = 'openai'
    samples_per_class = 50
    n_classes = 0
    min_samples = 4
    image_size = 336
    hidden_layer = 768
    seed = 5
    workers = 12
    train_batch_size = 16
    valid_batch_size = 32 
    emb_size = 768
    vit_bb_lr = {'10': 1.25e-6, '20': 2.5e-6, '26': 5e-6, '32': 10e-6} 
    vit_bb_wd = 1e-3
    hd_lr = 3e-4
    hd_wd = 1e-5
    autocast = True
    n_warmup_steps = 1000
    n_epochs = 2
    device = torch.device('cuda')
    s=30.
    m=.45
    m_min=.05
    acc_steps = 4
    global_step = 0
    reduce_lr = 0.1
    crit = 'ce'
    

def training(train_loader, 
             gallery_loader, 
             query_loader, 
             experiment_folder, 
             version='v1', 
             k=3, 
             reduce_lr_on_epoch=1,
             use_rampup=True):
    
    os.makedirs(experiment_folder, exist_ok=True)
    
    backbone, _, _ = open_clip.create_model_and_transforms(CFG.model_name, CFG.model_data)
    
    model = Model(backbone, CFG.hidden_layer, version, k).to(CFG.device)
    
#     model.half()
    
    optimizer = torch.optim.AdamW(model.get_parameters())
 
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)

    steps_per_epoch = math.ceil(len(train_loader) / CFG.acc_steps)

    num_training_steps = math.ceil(CFG.n_epochs * steps_per_epoch)
    
    if use_rampup:
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=num_training_steps,
                                                    num_warmup_steps=CFG.n_warmup_steps)  
    else:
        scheduler = get_constant_schedule(optimizer)
        
    best_score = 0
    best_updated_ = 0
    CFG.global_step = 0                   
    for epoch in range(math.ceil(CFG.n_epochs)):
        print(f'starting epoch {epoch}')

        # train of product-10k
        train(model, train_loader, optimizer, scaler, scheduler, epoch)

        # aicrowd test data
        print('gallery embeddings')
        embeddings_gallery, labels_gallery = val(model, gallery_loader)
        print('query embeddings')
        embeddings_query, labels_query = val(model, query_loader)

        # idk why it is needed
        gc.collect()
        torch.cuda.empty_cache() 

        # calculate validation score
        _, indices = utilities.get_similiarity_l2(embeddings_gallery, embeddings_query, 1000)


        indices = indices.tolist()
        labels_gallery = labels_gallery.tolist()
        labels_query = labels_query.tolist()

        preds = utilities.convert_indices_to_labels(indices, labels_gallery)
        score = utilities.map_per_set(labels_query, preds)
        print('validation score', score)

        # save model
        torch.save({
                'model_state_dict': model.encoder.state_dict(),
                }, f'{experiment_folder}/model_epoch_{epoch+1}_mAP3_{score:.2f}.pt')

        # early stopping
        if score > best_score:
            best_updated_ = 0
            best_score = score

        best_updated_ += 1

        if best_updated_ >= 3:
            print('no improvement done training....')
            return model
            
        if (epoch + 1) % reduce_lr_on_epoch == 0:
            scheduler.base_lrs = [g['lr'] * CFG.reduce_lr for g in optimizer.param_groups]
            
        # to speed up the training
        if epoch > 3:
            return model



def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    # aicrowd datasets
    gallery_loader = aicrowd_data_loader('.../products-10k/development_test_data/gallery.csv') 
    query_loader = aicrowd_data_loader('.../products-10k/development_test_data/queries.csv')

    k = 3  
    version = 'v2'
    data_aug = 'image_net'
    CFG.reduce_lr = 0.1
    train_loader = get_product_10k_dataloader(data_train, data_aug)
    experiment_folder = f'my_experiments/{CFG.model_name}-{CFG.model_data}-{str(data_aug)}-{str(version)}-p10k-h&m-Arcface(k={str(k)})-All-Epoch({str(CFG.n_epochs)})-Reduce_LR_0.1'
    model = training(train_loader, 
            gallery_loader, 
            query_loader, 
            experiment_folder, 
            version=version,
            k=k)
    # idk why it is needed
    gc.collect()
    torch.cuda.empty_cache()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to config file.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
