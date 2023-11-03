import glob
import sys
import os
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

# UTILS
# import vpr_utils as utilities

# %load_ext autoreload
# %autoreload 2


class Head(nn.Module):
    def __init__(self, hidden_size, k=3):
        super(Head, self).__init__()
        self.emb = nn.Linear(hidden_size, CFG.emb_size, bias=False)
        self.dropout = utilities.Multisample_Dropout()
        self.arc = utilities.ArcMarginProduct_subcenter(CFG.emb_size, CFG.n_classes, k)
        
    def forward(self, x):
        embeddings = self.dropout(x, self.emb)
        output = self.arc(embeddings)
        return output, F.normalize(embeddings)
    
class HeadV2(nn.Module):
    def __init__(self, hidden_size, k=3):
        super(HeadV2, self).__init__()
        self.arc = utilities.ArcMarginProduct_subcenter(hidden_size, CFG.n_classes, k)
        
    def forward(self, x):
        output = self.arc(x)
        return output, F.normalize(x)
    
class HeadV3(nn.Module):
    def __init__(self, hidden_size, k=3):
        super(HeadV3, self).__init__()        
        self.emb = nn.Linear(hidden_size, CFG.emb_size, bias=False)
        self.dropout = nn.Dropout1d(0.2)
        self.arc = utilities.ArcMarginProduct_subcenter(CFG.emb_size, CFG.n_classes, k)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.emb(x)
        output = self.arc(x)
        return output, F.normalize(x)



class Model(nn.Module):
    def __init__(self, vit_backbone, head_size, version='v1', k=3):
        super(Model, self).__init__()
        if version == 'v1':
            self.head = Head(head_size, k)
        elif version == 'v2':
            self.head = HeadV2(head_size, k)
        elif version == 'v3':
            self.head = HeadV3(head_size, k)
        else:
            self.head = Head(head_size, k)
        
        self.encoder = vit_backbone.visual
    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)

    def get_parameters(self):

        parameter_settings = [] 
        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.encoder.named_parameters()], 
                lr=CFG.vit_bb_lr, 
                wd=CFG.vit_bb_wd
            )
        ) 

        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.head.named_parameters()], 
                lr=CFG.hd_lr, 
                wd=CFG.hd_wd
            )
        ) 

        return parameter_settings

    def get_parameter_section(self, parameters, lr=None, wd=None): 
        parameter_settings = []


        lr_is_dict = isinstance(lr, dict)
        wd_is_dict = isinstance(wd, dict)

        layer_no = None
        for no, (n,p) in enumerate(parameters):
            
            for split in n.split('.'):
                if split.isnumeric():
                    layer_no = int(split)
            
            if not layer_no:
                layer_no = 0
            
            if lr_is_dict:
                for k,v in lr.items():
                    if layer_no < int(k):
                        temp_lr = v
                        break
            else:
                temp_lr = lr

            if wd_is_dict:
                for k,v in wd.items():
                    if layer_no < int(k):
                        temp_wd = v
                        break
            else:
                temp_wd = wd

            weight_decay = 0.0 if 'bias' in n else temp_wd

            parameter_setting = {"params" : p, "lr" : temp_lr, "weight_decay" : temp_wd}

            parameter_settings.append(parameter_setting)

            #print(f'no {no} | params {n} | lr {temp_lr} | weight_decay {weight_decay} | requires_grad {p.requires_grad}')

        return parameter_settings


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    num_classes = config.dataset.num_of_classes
    if arch.startswith('resnet'):
        model = models.__dict__[arch](weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception('model type is not supported:', arch)
    model.to('cuda')
    return model
