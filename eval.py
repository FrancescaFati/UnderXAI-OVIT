import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve,
    auc, 
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import wandb
import torch.nn.functional as F
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    LayerGradCam
)
import json
import logging

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score as sklearn_f1_score
import wandb
import logging
import engine 
from engine import log_attention_weights
import pandas as pd 
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import wandb

import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
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
            model: Neural network model
            device: Computation device (CPU/GPU)
            test_loader: Validation data loader
            checkpoint_path: Path to model checkpoint
            config: Configuration dictionary
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        self.test_loader = test_loader
        self.num_classes = config["num_classes"]
        #self.model = config["cnn"]
        
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
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
            all_preds, all_labels, all_probs = [], [], []
            all_logits = []
            all_gts = []

            self.model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(loader):
                    images = batch['image'].to(self.device)
                    #clinical_features = batch['clinical_features'].to(self.device)
                    #masks = batch['mask'].to(self.device)
                    
                    if self.num_classes == 1:
                        labels = batch["label"][:, 1].to(self.device)
                    else:
                        labels = batch["label"].to(self.device)
                    

                    logits, att_weights, _ = self.model(images)
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
                
                all_logits = torch.cat(all_logits, dim=0)
                all_gts = torch.cat(all_gts, dim=0)
                probabilities = torch.sigmoid(all_logits)
                preds = (probabilities >= 0.5).int()
                preds_opthr = (probabilities >= optimal_threshold).int()

                comprehensive_test_metrics = metrics.calculate_metrics(all_logits, all_gts, 0.5, split="Test")        
                comprehensive_test_metrics_opthr = metrics.calculate_metrics(all_logits, all_gts, optimal_threshold, split="Test_OPTHR")
                metrics.log_roc_curve(all_logits, all_gts, prefix="Test")
                
                wandb.log(comprehensive_test_metrics)
                wandb.log(comprehensive_test_metrics_opthr)       
                
                 
                df = pd.DataFrame({
                    'targets': all_gts.view(-1).cpu().numpy(),
                    'logits': all_logits.view(-1).cpu().numpy(),
                    'probabilities': probabilities.view(-1).cpu().numpy(), 
                    'preds': preds.view(-1).cpu().numpy(), 
                    'preds_opthr': preds_opthr.view(-1).cpu().numpy(),

                })
                df.to_excel(f"TEST_{wandb.run.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                    
            return comprehensive_test_metrics, comprehensive_test_metrics_opthr



    def evaluate_metrics(self) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        try:
            predictions, labels, probabilities = self.predict(self.test_loader, return_probs=True)
            
            # Convert to binary format if needed
            if len(labels.shape) > 1:
                labels = labels.argmax(axis=1)
            if len(predictions.shape) > 1:
                predictions = predictions.argmax(axis=1)
                
            metrics = {
                'accuracy': balanced_accuracy_score(labels, predictions),
                'f1_score': f1_score(labels, predictions, average='weighted', zero_division=0),
                'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
                'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
            }
            
            # Calculate ROC and PR curves if we have probabilities
            if probabilities.size > 0:
                # ROC curve and AUC
                fpr, tpr, _ = roc_curve(labels, probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities)
                metrics['roc_auc'] = auc(fpr, tpr)
                
                # Precision-Recall curve and AUC
                precision, recall, _ = precision_recall_curve(
                    labels, 
                    probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities
                )
                metrics['pr_auc'] = auc(recall, precision)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def print_classification_report(self) -> None:
        """Print detailed classification report."""
        try:
            predictions, labels, _ = self.predict(self.test_loader)
            
            if len(labels.shape) > 1:
                labels = labels.argmax(axis=1)
            if len(predictions.shape) > 1:
                predictions = predictions.argmax(axis=1)
                
            print("\nClassification Report:")
            print(classification_report(labels, predictions, zero_division=0))
            
        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
            raise


    def visualize_results(
        self,
        output_dir: str = 'evaluation_results'
    ) -> None:
        """
        Generate and save visualization plots.
        
        Args:
            output_dir: Directory to save visualization results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get predictions for visualization
            predictions, labels, probabilities = self.predict(self.test_loader, return_probs=True)
            
            # Generate different plots
            self._plot_roc_curve(labels, probabilities, output_path)
            self._plot_pr_curve(labels, probabilities, output_path)
            self._plot_confusion_matrix(labels, predictions, output_path)
            
            # Plot slice performance if we have 3D images
            if self.test_loader.dataset[0]['image'].dim() == 5:  # B, C, H, W, D
                slice_metrics = self.analyze_slice_performance()
                self._plot_slice_performance(slice_metrics, output_path)
            
            # Log to wandb if available
            if wandb.run is not None:
                for plot_path in output_path.glob('*.png'):
                    wandb.log({plot_path.stem: wandb.Image(str(plot_path))})
                    
            logger.info(f"Saved visualization plots to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def _plot_roc_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        output_dir: Path
    ) -> None:
        """Plot and save ROC curve."""
        try:
            plt.figure(figsize=(10, 8))
            
            # Convert to binary format if needed
            if len(labels.shape) > 1:
                labels = labels.argmax(axis=1)
            
            if len(probabilities.shape) > 1:
                prob_scores = probabilities[:, 1]
            else:
                prob_scores = probabilities
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(labels, prob_scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save
            plt.savefig(output_dir / 'roc_curve.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            raise

    def _plot_pr_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        output_dir: Path
    ) -> None:
        """Plot and save Precision-Recall curve."""
        try:
            plt.figure(figsize=(10, 8))
            
            # Convert to binary format if needed
            if len(labels.shape) > 1:
                labels = labels.argmax(axis=1)
            
            if len(probabilities.shape) > 1:
                prob_scores = probabilities[:, 1]
            else:
                prob_scores = probabilities
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(labels, prob_scores)
            pr_auc = auc(recall, precision)
            
            # Plot
            plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save
            plt.savefig(output_dir / 'pr_curve.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting PR curve: {str(e)}")
            raise

    def _plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        output_dir: Path
    ) -> None:
        """Plot and save confusion matrix."""
        try:
            plt.figure(figsize=(10, 8))
            
            # Convert to binary format if needed
            if len(labels.shape) > 1:
                labels = labels.argmax(axis=1)
            if len(predictions.shape) > 1:
                predictions = predictions.argmax(axis=1)
            
            # Calculate confusion matrix
            cm = confusion_matrix(labels, predictions)
            
            # Plot
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            # Save
            plt.savefig(output_dir / 'confusion_matrix.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise

    def analyze_slice_performance(self) -> Dict[str, Dict[int, float]]:
        """
        Analyze model performance by slice position for 3D images.
        
        Returns:
            Dictionary containing metrics per slice position
        """
        try:
            slice_metrics = defaultdict(lambda: defaultdict(list))
            
            with torch.no_grad():
                for batch in self.test_loader:
                    images = batch['image'].to(self.device)  # [B, C, H, W, D]
                    labels = batch['label'].to(self.device)
                    
                    # For each slice position
                    for slice_idx in range(images.shape[-1]):
                        # Get slice
                        slice_input = images[..., slice_idx]  # [B, C, H, W]
                        
                        # Get predictions
                        outputs = self.model(slice_input)
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()
                        
                        # Calculate metrics
                        accuracy = (preds == labels).float().mean().item()
                        f1 = f1_score(
                            labels.cpu().numpy(), 
                            preds.cpu().numpy(), 
                            average='weighted',
                            zero_division=0
                        )
                        
                        # Store metrics
                        slice_metrics['accuracy'][slice_idx].append(accuracy)
                        slice_metrics['f1_score'][slice_idx].append(f1)
            
            # Average metrics across batches
            return {
                metric: {
                    slice_idx: np.mean(values)
                    for slice_idx, values in slice_data.items()
                }
                for metric, slice_data in slice_metrics.items()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing slice performance: {str(e)}")
            raise

    def _plot_slice_performance(
        self,
        slice_metrics: Dict[str, Dict[int, float]],
        output_dir: Path
    ) -> None:
        """Plot and save slice-wise performance metrics."""
        try:
            plt.figure(figsize=(12, 6))
            
            for metric, values in slice_metrics.items():
                positions = list(values.keys())
                scores = list(values.values())
                plt.plot(positions, scores, '-o', label=metric.capitalize())
            
            plt.xlabel('Slice Position')
            plt.ylabel('Score')
            plt.title('Performance by Slice Position')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(output_dir / 'slice_performance.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting slice performance: {str(e)}")
            raise

    def analyze_errors(self, num_samples: int = 10) -> Dict[str, List]:
        """
        Analyze prediction errors.
        
        Args:
            num_samples: Number of error samples to analyze
            
        Returns:
            Dictionary containing error analysis
        """
        try:
            predictions, labels, probabilities = self.predict(self.test_loader, return_probs=True)
            
            # Convert to binary format if needed
            if len(labels.shape) > 1:
                labels = labels.argmax(axis=1)
            if len(predictions.shape) > 1:
                predictions = predictions.argmax(axis=1)
            
            # Find errors
            error_mask = predictions != labels
            error_indices = np.where(error_mask)[0]
            
            if len(error_indices) == 0:
                logger.info("No prediction errors found")
                return {}
            
            # Sample errors
            sample_size = min(num_samples, len(error_indices))
            sample_indices = np.random.choice(error_indices, size=sample_size, replace=False)
            
            error_analysis = {
                'indices': sample_indices.tolist(),
                'true_labels': labels[sample_indices].tolist(),
                'predicted_labels': predictions[sample_indices].tolist(),
                'confidences': probabilities[sample_indices].tolist()
            }
            
            return error_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing errors: {str(e)}")
            raise


    def save_final_model(
        self,
        save_dir: Path,
        predictions: np.ndarray,
        labels: np.ndarray,
        config: Dict,
        run_id: Optional[str] = None
    ) -> None:
        """
        Save the final model with its metrics.
        
        Args:
            save_dir: Directory to save the model
            predictions: Model predictions
            labels: True labels
            config: Model configuration
            run_id: Optional wandb run ID
        """
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate final metrics
            final_metrics = {
                'val_accuracy': (predictions == labels).mean(),
                'val_f1_score': binary_f1_score(
                    torch.tensor(predictions),
                    torch.tensor(labels),
                    zero_division=0
                ).item()
            }
            
            # Prepare model save path
            model_name = f"final_model_{run_id if run_id else 'default'}.pt"
            save_path = save_dir / model_name
            
            # Save model and metadata
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'final_metrics': final_metrics,
                'timestamp': wandb.run.start_time if wandb.run else None
            }, save_path)
            
            logger.info(f"Saved final model to {save_path}")
            logger.info(f"Final metrics: {final_metrics}")
            
            # Log to wandb if available
            if wandb.run:
                wandb.log({
                    "final_metrics": final_metrics,
                    "model_path": str(save_path)
                })
                
        except Exception as e:
            logger.error(f"Error saving final model: {str(e)}")
            raise
        

    def evaluate_by_slice_position(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model performance for each slice position.
        """
        try:
            slice_metrics = {}
            
            # Get number of slices from the first batch
            sample_batch = next(iter(data_loader))
            num_slices = sample_batch['image'].shape[-1]
            
            # Convert labels to binary format if they're one-hot
            if len(labels.shape) > 1:
                labels = labels.argmax(axis=1)
            if len(predictions.shape) > 1:
                predictions = predictions.argmax(axis=1)
            
            for slice_idx in range(num_slices):
                # Get predictions and labels for this slice
                slice_mask = self._get_slice_mask(data_loader, slice_idx)
                
                slice_preds = predictions[slice_mask]
                slice_labels = labels[slice_mask]
                
                # Calculate metrics
                if len(slice_preds) > 0:
                    accuracy = (slice_preds == slice_labels).mean()
                    f1 = sklearn_f1_score(
                        slice_labels,
                        slice_preds,
                        average='binary',
                        zero_division=0
                    )
                    
                    slice_metrics[slice_idx] = {
                        'accuracy': float(accuracy),
                        'f1_score': float(f1),
                        'sample_count': len(slice_preds)
                    }
                    
                    logger.debug(f"Slice {slice_idx} metrics: acc={accuracy:.3f}, f1={f1:.3f}")
            
            return slice_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating slice positions: {str(e)}")
            logger.error(f"Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")
            raise

    def plot_slice_importance(
        self,
        slice_metrics: Dict[int, Dict[str, float]],
        output_dir: str = 'evaluation_results'
    ) -> None:
        """
        Plot slice importance metrics.
        
        Args:
            slice_metrics: Dictionary mapping slice indices to their metrics
            output_dir: Directory to save visualization results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            
            # Extract metrics
            slices = list(slice_metrics.keys())
            accuracies = [metrics['accuracy'] for metrics in slice_metrics.values()]
            f1_scores = [metrics['f1_score'] for metrics in slice_metrics.values()]
            
            # Plot metrics
            plt.plot(slices, accuracies, 'b-', label='Accuracy', marker='o')
            plt.plot(slices, f1_scores, 'r-', label='F1 Score', marker='o')
            
            # Add mean lines
            mean_acc = np.mean(accuracies)
            mean_f1 = np.mean(f1_scores)
            plt.axhline(y=mean_acc, color='b', linestyle='--', alpha=0.5,
                       label=f'Mean Accuracy: {mean_acc:.3f}')
            plt.axhline(y=mean_f1, color='r', linestyle='--', alpha=0.5,
                       label=f'Mean F1: {mean_f1:.3f}')
            
            # Styling
            plt.xlabel('Slice Position')
            plt.ylabel('Score')
            plt.title('Slice-wise Performance Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value annotations
            for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
                plt.annotate(f'{acc:.2f}', (slices[i], acc), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
                plt.annotate(f'{f1:.2f}', (slices[i], f1), textcoords="offset points", 
                           xytext=(0,-15), ha='center', fontsize=8)
            
            # Save plot
            save_path = output_path / 'slice_importance.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({"slice_importance": wandb.Image(str(save_path))})
            
            plt.close()
            logger.info(f"Saved slice importance plot to {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting slice importance: {str(e)}")
            raise
    
    def _get_slice_mask(
        self,
        data_loader: torch.utils.data.DataLoader,
        slice_idx: int
    ) -> np.ndarray:
        """
        Get mask for a specific slice position.
        
        Args:
            data_loader: DataLoader containing the validation data
            slice_idx: Index of the slice to analyze
            
        Returns:
            Boolean mask array indicating which predictions correspond to the slice
        """
        mask = []
        for batch in data_loader:
            batch_size = batch['image'].shape[0]
            mask.extend([True] * batch_size)  # All predictions for this slice
        return np.array(mask)


    def plot_slice_importance(
        slice_metrics: Dict[int, Dict[str, float]],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot slice importance metrics.
        
        Args:
            slice_metrics: Dictionary mapping slice indices to their metrics
            save_path: Optional path to save the plot
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Extract metrics
            slices = list(slice_metrics.keys())
            accuracies = [metrics['accuracy'] for metrics in slice_metrics.values()]
            f1_scores = [metrics['f1_score'] for metrics in slice_metrics.values()]
            
            # Plot metrics
            plt.plot(slices, accuracies, 'b-', label='Accuracy', marker='o')
            plt.plot(slices, f1_scores, 'r-', label='F1 Score', marker='o')
            
            # Add mean lines
            mean_acc = np.mean(accuracies)
            mean_f1 = np.mean(f1_scores)
            plt.axhline(y=mean_acc, color='b', linestyle='--', alpha=0.5,
                    label=f'Mean Accuracy: {mean_acc:.3f}')
            plt.axhline(y=mean_f1, color='r', linestyle='--', alpha=0.5,
                    label=f'Mean F1: {mean_f1:.3f}')
            
            # Styling
            plt.xlabel('Slice Position')
            plt.ylabel('Score')
            plt.title('Slice-wise Performance Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value annotations
            for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
                plt.annotate(f'{acc:.2f}', (slices[i], acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
                plt.annotate(f'{f1:.2f}', (slices[i], f1), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=8)
            
            # Save if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path / 'slice_importance.png', bbox_inches='tight', dpi=300)
                if wandb.run:
                    wandb.log({"slice_importance": wandb.Image(str(save_path / 'slice_importance.png'))})
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting slice importance: {str(e)}")
            raise

class ExplanationVisualizer:
    """Class for visualization of model explanations and analysis"""
    
    @staticmethod
    def plot_slice_importance(
        slice_metrics: Dict[int, Dict[str, float]],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot slice importance metrics.
        
        Args:
            slice_metrics: Dictionary mapping slice indices to their metrics
            save_path: Optional path to save the plot
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Extract metrics
            slices = list(slice_metrics.keys())
            accuracies = [metrics['accuracy'] for metrics in slice_metrics.values()]
            f1_scores = [metrics['f1_score'] for metrics in slice_metrics.values()]
            
            # Plot metrics
            plt.plot(slices, accuracies, 'b-', label='Accuracy', marker='o')
            plt.plot(slices, f1_scores, 'r-', label='F1 Score', marker='o')
            
            # Add mean line
            mean_acc = np.mean(accuracies)
            plt.axhline(y=mean_acc, color='b', linestyle='--', alpha=0.5,
                       label=f'Mean Accuracy: {mean_acc:.3f}')
            
            # Styling
            plt.xlabel('Slice Position')
            plt.ylabel('Score')
            plt.title('Slice-wise Performance Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value annotations
            for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
                plt.annotate(f'{acc:.2f}', (slices[i], acc), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
                plt.annotate(f'{f1:.2f}', (slices[i], f1), textcoords="offset points", 
                           xytext=(0,-15), ha='center', fontsize=8)
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path / 'slice_importance.png', bbox_inches='tight', dpi=300)
                if wandb.run:
                    wandb.log({"slice_importance": wandb.Image(str(save_path / 'slice_importance.png'))})
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting slice importance: {str(e)}")
            raise