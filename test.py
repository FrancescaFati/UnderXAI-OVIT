"""
Test script for evaluating a trained model on a test dataset with bootstrapping.
- Loads a model checkpoint and configuration.
- Runs predictions and computes metrics using bootstrapped samples.
- Saves results to CSV files for further analysis.
- Designed for sharing in a public repository (edit hardcoded paths as needed).
"""
import torch
import os
import yaml
import wandb
from rich import print
from datetime import datetime
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from model import Vit_Classifier
from pcs_dataset import PCSDataset
from utilis import detect_aval_cpus
import metrics

# --- User configuration: Edit these paths for your environment ---
TEST_DATA_PATH = ""   # Path to test dataset JSON
CONFIG_YAML_PATH = "" # Path to model config YAML
CHECKPOINT_PATH = ""  # Path to model checkpoint
# --------------------------------------------------------------

# Prepare test dataset and loader
# (Edit PCSDataset arguments as needed for your project)
test_dataset = PCSDataset(
    data_file=TEST_DATA_PATH,
    transform=None,
    is_training=False,
    seg=False,
    seg_as_im=True,
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

# Load model configuration from YAML
with open(CONFIG_YAML_PATH, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

# Instantiate and load model checkpoint
model = Vit_Classifier(config).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

def predict(loader):
    """
    Run model inference on a DataLoader and compute metrics.
    Args:
        loader (DataLoader): DataLoader for evaluation.
    Returns:
        tuple: (comprehensive_test_metrics, all_predictions_dict)
    """
    all_logits = []
    all_gts = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch['image'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            labels = batch["label"][:, 1].to(device)
            logits, _, _ = model(images, clinical_features)
            all_logits.append(logits.detach())
            all_gts.append(labels.detach())
        all_logits = torch.cat(all_logits, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        probabilities = torch.sigmoid(all_logits)
        preds = (probabilities >= 0.5).int().view(-1)
        comprehensive_test_metrics = metrics.calculate_metrics(all_logits, all_gts, 0.5, split="Test")
        alls = {
            "logits": all_logits.cpu().numpy(),
            "gts": all_gts.cpu().numpy(),
            "preds": preds.cpu().numpy(),
        }
    return comprehensive_test_metrics, alls

# --- Bootstrapping for robust metric estimation ---
n_iterations = 1000
n_size = len(test_dataset)
bootstrap_metrics = []
bootstrap_alls = []

for i in range(n_iterations):
    # Sample with replacement to create a bootstrap subset
    indices = np.random.choice(np.arange(n_size), size=n_size, replace=True)
    bootstrap_subset = Subset(test_dataset, indices)
    bootstrap_loader = DataLoader(bootstrap_subset, batch_size=1, shuffle=False)
    comprehensive_test_metrics, alls = predict(bootstrap_loader)
    bootstrap_metrics.append(comprehensive_test_metrics)
    bootstrap_alls.append(alls)
    print(f"{i+1} / {n_iterations}")

# Save bootstrapped metrics and predictions to CSV files
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
pd.DataFrame(bootstrap_metrics).to_csv(f"METRICS_1000_bootstrap_{timestamp}.csv", index=False)
pd.DataFrame(bootstrap_alls).to_csv(f"ALLS_1000_bootstrap_{timestamp}.csv", index=False)
