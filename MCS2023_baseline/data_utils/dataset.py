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

class SubmissionDataset(Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)
        product_id = self.imlist['product_id'][index]
        return img, product_id

    def __len__(self):
        return len(self.imlist)

        
def read_img(img_path, is_gray=False):
    mode = cv2.IMREAD_COLOR if not is_gray else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(img_path, mode)
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_final_transform():  
    final_transform = T.Compose([
            T.Resize(
                size=(CFG.image_size, CFG.image_size), 
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True),
            T.ToTensor(), 
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
    return final_transform

class ProductDataset(Dataset):
    def __init__(self, 
                 data, 
                 transform=None, 
                 final_transform=None):
        self.data = data
        self.transform = transform
        self.final_transform = final_transform
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
       
        img = read_img(self.data[idx][0])            
        
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                img = self.transform(image=img)['image']
            else:
                img = self.transform(img)
        
        if self.final_transform is not None:
            if isinstance(img, np.ndarray):
                img =  Image.fromarray(img)
            img = self.final_transform(img)
            
        product_id = self.data[idx][1]
        return {"images": img, "labels": product_id}
    
def get_product_10k_dataloader(data_train, data_aug='image_net'):
    
    transform = None
    if data_aug == 'image_net':
        transform = T.Compose([
            T.ToPILImage(),
            T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)
        ])
        
    elif data_aug == 'aug_mix':
        transform = T.Compose([
            T.ToPILImage(),
            T.AugMix()
        ])
    elif data_aug == 'happy_whale':
        aug8p3 = A.OneOf([
            A.Sharpen(p=0.3),
            A.ToGray(p=0.3),
            A.CLAHE(p=0.3),
        ], p=0.5)

        transform = A.Compose([
            A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.1, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.Resize(CFG.image_size, CFG.image_size),
            aug8p3,
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ])
    
    elif data_aug == 'cut_out':        
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
            A.Resize(CFG.image_size, CFG.image_size),
            A.Cutout(max_h_size=int(CFG.image_size * 0.4), 
                     max_w_size=int(CFG.image_size * 0.4), 
                     num_holes=1, p=0.5),
        ])
    elif data_aug == 'clip':
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(
                size=(224, 224), 
                scale=(0.9, 1.0), 
                ratio=(0.75, 1.3333), 
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True
            )
        ])
    elif data_aug == 'clip+image_net':
        transform = T.Compose([
            T.ToPILImage(),
            T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
            T.RandomResizedCrop(
                size=(224, 224), 
                scale=(0.9, 1.0), 
                ratio=(0.75, 1.3333), 
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True
            )
        ])
    
    final_transform = get_final_transform()
    train_dataset = ProductDataset(data_train, 
                                   transform, 
                                   final_transform)
    train_loader = DataLoader(train_dataset, 
                              batch_size = CFG.train_batch_size, 
                              num_workers=CFG.workers, 
                              shuffle=True, 
                              drop_last=True)
    print(f'Training Data -> Dataset Length ({len(train_dataset)})')
    return train_loader

def aicrowd_data_loader(csv_path, img_dir='/kaggle/input/products-10k/products-10k/development_test_data'):
    df_g = pd.read_csv(csv_path)
    df_g_ = df_g[['img_path', 'product_id']]
    df_g_['img_path'] = df_g_.apply(lambda x: img_dir + '/' + x['img_path'], axis=1)
    data_ = np.array(df_g_).tolist()
    
    final_transform = get_final_transform()
    dataset = ProductDataset(data_, None, final_transform)
    data_loader = DataLoader(dataset, 
                             batch_size = CFG.valid_batch_size, 
                             num_workers=CFG.workers, 
                             shuffle=False, 
                             drop_last=False)
    
    print(f'{csv_path} -> Dataset Length ({len(dataset)})')
    return data_loader