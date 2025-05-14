import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import DataLoader
import wandb
import torch.nn.functional as F
import logging
import pandas as pd 
from datetime import datetime
import metrics
from engine import log_attention_weights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates a trained model on a given dataset and computes relevant metrics.
    Handles loading from checkpoint, device management, and logging.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        test_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the model evaluator.
        Args:
            model (nn.Module): Neural network model to evaluate.
            device (torch.device): Computation device (CPU/GPU).
            test_loader (DataLoader, optional): DataLoader for evaluation.
            checkpoint_path (str, optional): Path to model checkpoint.
            config (dict, optional): Configuration dictionary.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        self.test_loader = test_loader
        self.num_classes = config["num_classes"]
        try:
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            self.model = model.to(device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error initializing ModelEvaluator: {str(e)}")
            raise

    def predict(
        self,
        loader: DataLoader,
        optimal_threshold: float = 0.5,
    ) -> Dict:
        """
        Run model inference on the provided DataLoader and compute evaluation metrics.
        Args:
            loader (DataLoader): DataLoader for evaluation (e.g., test set).
            optimal_threshold (float): Threshold for binary classification.
        Returns:
            dict: Comprehensive test metrics as computed by the metrics module.
        """
        all_preds, all_labels, all_probs = [], [], []
        all_logits = []
        all_gts = []
        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                images = batch['image'].to(self.device)
                clinical_features = batch['clinical_features'].to(self.device)
                # Handle binary and multi-class cases
                if self.num_classes == 1:
                    labels = batch["label"][:, 1].to(self.device)
                else:
                    labels = batch["label"].to(self.device)
                logits, att_weights, _ = self.model(images, clinical_features)
                log_attention_weights(att_weights, step=batch_index, prefix="Test")
                if self.num_classes == 1:
                    probs = torch.sigmoid(logits)
                    binary_preds = (probs.squeeze() > 0.5).float()
                    binary_targets = labels
                else:
                    probs = torch.sigmoid(logits)
                    binary_preds = logits.argmax(dim=1)
                    binary_targets = labels.argmax(dim=1)
                all_preds.append(binary_preds.cpu().numpy().reshape(-1))
                all_labels.append(binary_targets.cpu().numpy().reshape(-1))
                all_probs.append(probs.cpu().numpy().reshape(probs.shape[0], -1))
                all_logits.append(logits.detach()) 
                all_gts.append(labels.detach())
            # Concatenate all predictions and ground truths
            all_logits = torch.cat(all_logits, dim=0)
            all_gts = torch.cat(all_gts, dim=0)
            probabilities = torch.sigmoid(all_logits)
            preds = (probabilities >= 0.5).int()
            preds_opthr = (probabilities >= optimal_threshold).int()
            # Compute and log comprehensive test metrics
            comprehensive_test_metrics = metrics.calculate_metrics(all_logits, all_gts, 0.5, split="Test")        
            metrics.log_roc_curve(all_logits, all_gts, prefix="Test")
            wandb.log(comprehensive_test_metrics) 
            # Save predictions and probabilities to Excel for further analysis
            df = pd.DataFrame({
                'targets': all_gts.view(-1).cpu().numpy(),
                'logits': all_logits.view(-1).cpu().numpy(),
                'probabilities': probabilities.view(-1).cpu().numpy(), 
                'preds': preds.view(-1).cpu().numpy(), 
                'preds_opthr': preds_opthr.view(-1).cpu().numpy(),
            })
            df.to_excel(f"TEST_{wandb.run.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        return comprehensive_test_metrics

