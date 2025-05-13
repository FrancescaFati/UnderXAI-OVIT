import torch
import os
import yaml
import wandb
from rich import print
from datetime import datetime

from monai.data import Dataset, ImageDataset, DataLoader

from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    Resize,
)
from monai.transforms import Compose, Lambda, ToTensor
import utilis
import models
import nibabel as nib
import pandas as pd
import numpy as np
import dill
import ast
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from model import Vit_Classifier 
from ds import PCSDataset
from utilis import detect_aval_cpus
import metrics
from torch.utils.data import DataLoader, Subset
import monai.networks.nets as nets


test_dataset = PCSDataset(
    data_file="/home/ffati/DATA/test_dataset_ids.json",
    excluded_years=[2015],
    transform=None,
    is_training=False, 
    seg = False, 
    im_bucket = False, 
    seg_bucket = False, 
    return_2d_slices = False, 
    r0r1_r2  = False,
    im_masked  = False,  
    contrastive = False, 
    seg_as_im = False,
    clinical_fe=True,
)  


test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=detect_aval_cpus(),
    pin_memory=True,
    drop_last=False
)    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_yaml = '/home/ffati/UnderXAI/config.yaml'

with open(config_yaml, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)
    
model = Vit_Classifier(config).to(device)

checkpoint =torch.load("/home/ffati/UnderXAI/models/final_model_dkrbrhk6.pt", map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])

def predict(loader):
    all_preds, all_labels, all_probs = [], [], []
    all_logits = []
    all_gts = []
    clinical_fes = []

    model.eval()
    with torch.no_grad():

        for batch_index, batch in enumerate(loader):
            images = batch['image'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            labels = batch["label"][:, 1].to(device)

            logits, _ ,_ = model(images, clinical_features)

            probs = torch.sigmoid(logits)
            binary_preds = (probs.squeeze() > 0.5).float()
            binary_targets = labels

            all_preds.append(binary_preds.cpu().numpy().reshape(-1))
            all_labels.append(binary_targets.cpu().numpy().reshape(-1))
            all_probs.append(probs.cpu().numpy().reshape(probs.shape[0], -1))

            all_logits.append(logits.detach())
            all_gts.append(labels.detach())
            #clinical_fes.append(clinical_features.cpu().numpy())

        all_logits = torch.cat(all_logits, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        probabilities = torch.sigmoid(all_logits)
        preds = (probabilities >= 0.5).int().view(-1)

        #clinical_fes = np.concatenate(clinical_fes, axis=0)

        comprehensive_test_metrics = metrics.calculate_metrics(all_logits, all_gts, 0.5, split="Test")
        alls = {
            #"logits": all_logits.cpu().numpy(),
            "gts": all_gts.cpu().numpy(),
            "preds": preds.cpu().numpy(),
            #"probabilities": probabilities.cpu().numpy(),
            #"clinical_fe": clinical_fes
        }

    return comprehensive_test_metrics, alls



n_iterations = 1000
n_size = len(test_dataset)
boostrap_metrics = []
boostrap_alls = []

    
for i in range(n_iterations): 
    
    indices =  np.random.choice(np.arange(n_size), size=n_size, replace=True)
    bootstrap_subset = Subset(test_dataset, indices)
    bootstrap_loader = DataLoader(bootstrap_subset, batch_size=1, shuffle=False)
    
    comprehensive_test_metrics, alls = predict(bootstrap_loader)
    boostrap_metrics.append(comprehensive_test_metrics)
    boostrap_alls.append(alls)
    print(f"{i+1} / {n_iterations}")
        
bootstrap_df_metrics = pd.DataFrame(boostrap_metrics)
bootstrap_df_alls = pd.DataFrame(boostrap_alls)

bootstrap_df_metrics.to_csv(f"METRICS_1000_bootstrap_dkrbrhk6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
bootstrap_df_alls.to_csv(f"ALLS_1000_bootstrap_dkrbrhk6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
