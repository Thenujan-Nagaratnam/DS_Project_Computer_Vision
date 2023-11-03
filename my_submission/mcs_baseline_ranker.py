import numpy as np

import yaml

import numpy as np
import torch
import torchvision.models as models

from collections import OrderedDict

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm

import sys
sys.path.append('./MCS2023_baseline/')

from data_utils.augmentations import get_val_aug
from data_utils.dataset import SubmissionDataset
from utils import convert_dict_to_tuple

class MCS_BaseLine_Ranker:
    def __init__(self, dataset_path, gallery_csv_path, queries_csv_path):
        """
        Initialize your model here
        Inputs:
            dataset_path
            gallery_csv_path
            queries_csv_path
        """

        self.device = 'cuda'
        self.batch_size = 512
      
        # Add your code below

        checkpoint_path = './MCS2023_baseline/experiments/baseline_mcs/baseline_model.pth'

        print('Creating model and loading checkpoint')
        self.model = models.__dict__[self.exp_cfg.model.arch](
            num_classes=self.exp_cfg.dataset.num_of_classes
        )
        checkpoint = torch.load(checkpoint_path,
                                map_location='cuda')['state_dict']

        vit_backbone = open_clip.create_model_and_transforms('ViT-L-14-336', None)[0].visual
        vit_backbone.load_state_dict(th.load(checkpoint_path))
        vit_backbone.half()   # Apply half precision to the backbone model
        vit_backbone.eval()   # Dropping unecessary layers
        
        self.model = vit_backbone

        print('Weights are loaded')


    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def get_transform():  
    transform = T.Compose([
            T.Resize(
                size=(224, 224), 
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True),
            T.ToTensor(), 
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
    return transform

    def predict_product_ranks(self):
        """
        This function should return a numpy array of shape `(num_queries, 1000)`. 
        For ach query image your model will need to predict 
        a set of 1000 unique gallery indexes, in order of best match first.

        Outputs:
            class_ranks - A 2D numpy array where the axes correspond to:
                          axis 0 - Batch size
                          axis 1 - An ordered rank list of matched image indexes, most confident prediction first
                            - maximum length of this should be 1000
                            - predictions above this limit will be dropped
                            - duplicates will be dropped such that the lowest index entry is preserved
        """

        
        img_dir = "/kaggle/input/vprtestdata/public_dataset/"
        
        transform = get_transform()

        dataset_train = SubmissionDataset(img_dir, os.path.join(img_dir, "gallery.csv"), transform)
        gallery_loader = DataLoader(dataset_train, batch_size=512, num_workers=4)
        dataset_test = SubmissionDataset(img_dir, os.path.join(img_dir, "queries.csv"), transform, with_bbox=True)
        query_loader = DataLoader(dataset_test, batch_size=512, num_workers=4)

        print('Calculating embeddings')
        gallery_embeddings = np.zeros((len(gallery_dataset), self.embedding_shape))
        query_embeddings = np.zeros((len(query_dataset), self.embedding_shape))

        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = outputs.data.cpu().numpy()
                gallery_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
            
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = outputs.data.cpu().numpy()
                query_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
        
        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)
        distances = pairwise_distances(query_embeddings, gallery_embeddings)
        sorted_distances = np.argsort(distances, axis=1)[:, :1000]

        class_ranks = sorted_distances
        return class_ranks