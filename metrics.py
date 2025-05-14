import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
import wandb

def log_roc_curve(all_logits, all_gts, prefix=""):
    """
    Calculate ROC curve, AUC, and log them to wandb using a custom implementation
    that avoids issues with wandb's built-in ROC curve function.

    Args:
        all_logits (torch.Tensor): Model logits (raw outputs before sigmoid)
        all_gts (torch.Tensor): Ground truth labels (0 or 1)
        prefix (str, optional): Prefix for metric names. Defaults to "".

    Returns:
        float: The ROC AUC score
    """
    # Ensure tensors are on CPU and convert to numpy
    all_logits = all_logits.detach().cpu()
    all_gts = all_gts.detach().cpu()

    # Calculate probabilities from logits
    probabilities = torch.sigmoid(all_logits)

    # Convert to numpy and flatten if needed
    if probabilities.ndim > 1:
        probs_np = probabilities.flatten().numpy()
    else:
        probs_np = probabilities.numpy()
    if all_gts.ndim > 1:
        gts_np = all_gts.flatten().numpy()
    else:
        gts_np = all_gts.numpy()

    # Calculate ROC curve using sklearn
    fpr, tpr, thresholds = roc_curve(gts_np, probs_np)
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Create a custom wandb Table for the ROC curve
    table = wandb.Table(columns=["False Positive Rate", "True Positive Rate", "Threshold"])
    for i in range(len(fpr)):
        table.add_data(float(fpr[i]), float(tpr[i]), float(thresholds[i]) if i < len(thresholds) else float('nan'))

    # Log the table and create a custom plot
    wandb.log({
        f"{prefix}roc_auc": roc_auc,
        f"{prefix}roc_curve": wandb.plot.line(
            table,
            "False Positive Rate",
            "True Positive Rate",
            title=f"ROC Curve (AUC = {roc_auc:.4f})"
        )
    })
    return roc_auc

def calculate_metrics(
    logits: torch.Tensor, ground_truth: torch.Tensor, threshold: float = 0.5, split=None
) -> dict:
    """
    Calculate a suite of metrics for binary classification, including accuracy, precision,
    recall, F1 score, ROC AUC, PR AUC, confusion matrix, and Cohen's Kappa.

    Args:
        logits (torch.Tensor): Tensor of shape (B, 1) containing model logits.
        ground_truth (torch.Tensor): Tensor of shape (B, 1) containing ground truth (0 or 1).
        threshold (float): Threshold for converting logits to binary predictions.
        split (str): A string identifier for the data split (e.g., 'train', 'val', 'test').

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1 score, ROC AUC, PR AUC, confusion matrix, and Cohen's Kappa.
    """
    # Convert logits to probabilities using the sigmoid function
    probabilities = torch.sigmoid(logits)
    # Convert probabilities to binary predictions using the threshold
    predictions = (probabilities >= threshold).int()
    # Flatten tensors for compatibility with metric calculations
    predictions = predictions.view(-1).cpu().numpy()
    ground_truth = ground_truth.view(-1).cpu().numpy()
    probabilities = probabilities.view(-1).cpu().numpy()

    # Calculate confusion matrix components
    TP = ((predictions == 1) & (ground_truth == 1)).sum()
    TN = ((predictions == 0) & (ground_truth == 0)).sum()
    FP = ((predictions == 1) & (ground_truth == 0)).sum()
    FN = ((predictions == 0) & (ground_truth == 1)).sum()

    # Core metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Cohen's Kappa Calculation
    cohen_kappa = (
        2 * (TP * TN - FP * FN) / ((TP + FP) * (FP + TN) + (TP + FN) * (FN + TN))
    )

    # Additional metrics (only if both classes are present)
    roc_auc = (
        roc_auc_score(ground_truth, probabilities)
        if len(set(ground_truth)) > 1
        else None
    )
    pr_auc = (
        average_precision_score(ground_truth, probabilities)
        if len(set(ground_truth)) > 1
        else None
    )

    # Confusion matrix as a dictionary for better interpretability
    confusion_matrix = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

    # Prefix metrics with split name if provided
    prefix = f"{split}/" if split is not None else ""
    return {
        f"{prefix}accuracy": accuracy,
        f"{prefix}precision": precision,
        f"{prefix}recall": recall,
        f"{prefix}f1_score": f1_score,
        f"{prefix}roc_auc": roc_auc,
        f"{prefix}pr_auc": pr_auc,
        f"{prefix}cohen_kappa": cohen_kappa,
        f"{prefix}confusion_matrix": confusion_matrix,
    }



