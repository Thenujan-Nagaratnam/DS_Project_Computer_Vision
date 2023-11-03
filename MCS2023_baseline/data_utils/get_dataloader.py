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
from dataset import get_product_10k_dataloader
# UTILS
# import vpr_utils as utilities

# %load_ext autoreload
# %autoreload 2

def get_dataloader():
    # used for training
    training_samples = []
    values_counts = []
    num_classes = 0

    # H&M
    files = glob.glob("../H&M/images/*/*")
    file_paths = dict((os.path.splitext(os.path.split(f)[-1])[0], f) for f in files)

    df = pd.read_csv('../H&M/articles.csv', 
                    usecols=['article_id', 'product_code'],
                    dtype={'article_id': str, 'product_code': str})

    groupped_products = {}
    for index, row in df.iterrows():
        v = groupped_products.get(row['product_code'], [])
        f = file_paths.get(row['article_id'])
        if f:
            groupped_products[row['product_code']] = v + [f]


    for key, value in groupped_products.items():
        if len(value) >= CFG.min_samples:
            paths = value[:CFG.samples_per_class]
            
            values_counts.append(len(paths))
            training_samples.extend([
                (p, num_classes) for p in paths
            ])
            num_classes += 1

            
    # Product-10k
    df = pd.read_csv('../Product10K/train.csv')
    df_g = df.groupby('class', group_keys=True).apply(lambda x: x)


    train_df = pd.read_csv('../Product10K/train.csv')
    train_df['path'] = train_df.apply(lambda x: '.../Product10K/train/' + '/' + x['name'], axis=1)


    # remove ../products-10k/test/9397815.jpg from the list!
    test_df = pd.read_csv('.../Product10K/test_kaggletest.csv')
    test_df = test_df.drop(test_df[test_df.name == '9397815.jpg'].index) # smt wrong with this img
    test_df['path'] = test_df.apply(lambda x: '.../Product10K/test/' + '/' + x['name'], axis=1)

    df = pd.concat([
        test_df[['class','path']],
        train_df[['class', 'path']]
    ])
    df_g = df.groupby('class', group_keys=True).apply(lambda x: x)


    for group in tqdm(set(df_g['class'])):
        names = list(df_g.path[df_g['class'] == group])
        if len(names) >= CFG.min_samples:
            paths = [
                name for name in names[:CFG.samples_per_class]
            ]

            values_counts.append(len(paths))
            training_samples.extend([
                (p, num_classes) for p in paths
            ])
            
            num_classes += 1

    data_train = training_samples 
    value_counts = np.array(values_counts)
    CFG.n_classes = num_classes

    data_aug = 'image_net'
    train_loader = get_product_10k_dataloader(data_train, data_aug)

    return train_loader