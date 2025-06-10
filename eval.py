import torch
import torch.nn as nn
from typing import Optional, Dict
from torch.utils.data import DataLoader
import logging
import pandas as pd 
from datetime import datetime
import metrics
from engine import log_attention_weights
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import wandb
    _wandb_available = True
except ImportError:
    wandb = None
    _wandb_available = False

class ModelEvaluator:
    """
    Evaluates a trained model on a given dataset and computes relevant metrics.
    Handles loading from checkpoint, device management, and optional experiment tracking.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        test_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict] = None,
        enable_wandb: bool = True
    ):
        """
        Initialize the model evaluator.
        Args:
            model (nn.Module): Neural network model to evaluate.
            device (torch.device): Computation device (CPU/GPU).
            test_loader (DataLoader, optional): DataLoader for evaluation.
            checkpoint_path (str, optional): Path to model checkpoint.
            config (dict, optional): Configuration dictionary.
            enable_wandb (bool): Whether to enable experiment tracking with wandb.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        self.test_loader = test_loader
        self.num_classes = self.config.get("num_classes", 1)
        self.enable_wandb = enable_wandb and _wandb_available
        if enable_wandb and not _wandb_available:
            logger.warning("wandb logging is enabled but wandb is not installed. Logging will be skipped.")
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
        all_logits = []
        all_gts = []

        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                images = batch['image'].to(self.device)
                clinical_features = batch['clinical_features'].to(self.device)
                labels = batch["label"][:, 1].to(self.device)

                logits, att_weights, _ = self.model(images, clinical_features)

                if att_weights is not None:
                    log_attention_weights(att_weights, step=batch_index, prefix="Test")

                probs = torch.sigmoid(logits)
                binary_preds = (probs.squeeze() > 0.5).float()
                binary_targets = labels
 
                all_logits.append(logits.detach()) 
                all_gts.append(labels.detach())
            # Concatenate all predictions and ground truths
            all_logits = torch.cat(all_logits, dim=0)
            all_gts = torch.cat(all_gts, dim=0)
            probabilities = torch.sigmoid(all_logits)
            preds = (probabilities >= 0.5).int()
            # Compute and log comprehensive test metrics
            comprehensive_test_metrics = metrics.calculate_metrics(all_logits, all_gts, 0.5, split="Test")        
            metrics.log_roc_curve(all_logits, all_gts, prefix="Test/")
            # Optionally log to wandb if enabled and available
            if self.enable_wandb:
                wandb.log(comprehensive_test_metrics)
            # Save predictions and probabilities to Excel for further analysis
            df = pd.DataFrame({
                'targets': all_gts.view(-1).cpu().numpy(),
                'logits': all_logits.view(-1).cpu().numpy(),
                'probabilities': probabilities.view(-1).cpu().numpy(), 
                'preds': preds.view(-1).cpu().numpy(), 
            })
            # Save with a generic timestamped filename
            save_dir = Path('test').absolute()
            save_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(f"test/test_results_{wandb.run.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        return comprehensive_test_metrics

