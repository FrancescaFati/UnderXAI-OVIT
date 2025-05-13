import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import wandb
    
def log_roc_curve(all_logits, all_gts, prefix=""):
    """
    Calculate ROC curve, AUC, and log them to wandb using a custom implementation
    that avoids issues with wandb's built-in ROC curve function.
    
    Args:
        all_logits (torch.Tensor): Model logits
        all_gts (torch.Tensor): Ground truth labels (0 or 1)
        prefix (str, optional): Prefix for metric names. Defaults to "".
    
    Returns:
        float: The ROC AUC score
    """
    # Ensure tensors are on CPU and convert to numpy
    all_logits = all_logits.detach().cpu()
    all_gts = all_gts.detach().cpu()
    
    # Calculate probabilities
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
    probabilities: torch.Tensor, ground_truth: torch.Tensor, threshold: float = 0.5, split=None
) -> dict:
    """
    Calculate metrics for binary classification, including Cohen's Kappa.

    Args:
        logits (torch.Tensor): Tensor of shape (B, 1) containing model logits.
        ground_truth (torch.Tensor): Tensor of shape (B, 1) containing ground truth (0 or 1).
        threshold (float): Threshold for converting logits to binary predictions.
        split (str): A string identifier for the data split (e.g., 'train', 'val', 'test').

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1 score, ROC AUC, PR AUC, confusion matrix, and Cohen's Kappa.
    """
    # Convert logits to probabilities using the sigmoid function
    #probabilities = torch.sigmoid(logits)

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

    # Metrics
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

    # Additional Metrics
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

    return {
        f"{split+'/' if split is not None else ''}accuracy": accuracy,
        f"{split+'/' if split is not None else ''}precision": precision,
        f"{split+'/' if split is not None else ''}recall": recall,
        f"{split+'/' if split is not None else ''}f1_score": f1_score,
        f"{split+'/' if split is not None else ''}roc_auc": roc_auc,
        f"{split+'/' if split is not None else ''}pr_auc": pr_auc,
        f"{split+'/' if split is not None else ''}cohen_kappa": cohen_kappa,
        f"{split+'/' if split is not None else ''}confusion_matrix": confusion_matrix,
    }

def calculate_metrics_probs(
    probs: torch.Tensor, ground_truth: torch.Tensor, threshold: float = 0.5, split=None
) -> dict:
    """
    Calculate metrics for binary classification, including Cohen's Kappa.

    Args:
        logits (torch.Tensor): Tensor of shape (B, 1) containing model logits.
        ground_truth (torch.Tensor): Tensor of shape (B, 1) containing ground truth (0 or 1).
        threshold (float): Threshold for converting logits to binary predictions.
        split (str): A string identifier for the data split (e.g., 'train', 'val', 'test').

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1 score, ROC AUC, PR AUC, confusion matrix, and Cohen's Kappa.
    """
    predictions = probs

    # Flatten tensors for compatibility with metric calculations
    predictions = predictions.view(-1).cpu().numpy()
    ground_truth = ground_truth.view(-1).cpu().numpy()
    probabilities = probs.view(-1).cpu().numpy()

    # Calculate confusion matrix components
    TP = ((predictions == 1) & (ground_truth == 1)).sum()
    TN = ((predictions == 0) & (ground_truth == 0)).sum()
    FP = ((predictions == 1) & (ground_truth == 0)).sum()
    FN = ((predictions == 0) & (ground_truth == 1)).sum()

    # Metrics
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

    # Additional Metrics
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

    return {
        f"{split+'/' if split is not None else ''}accuracy": accuracy,
        f"{split+'/' if split is not None else ''}precision": precision,
        f"{split+'/' if split is not None else ''}recall": recall,
        f"{split+'/' if split is not None else ''}f1_score": f1_score,
        f"{split+'/' if split is not None else ''}roc_auc": roc_auc,
        f"{split+'/' if split is not None else ''}pr_auc": pr_auc,
        f"{split+'/' if split is not None else ''}cohen_kappa": cohen_kappa,
        f"{split+'/' if split is not None else ''}confusion_matrix": confusion_matrix,
    }

def find_optimal_threshold(
    logits: torch.Tensor, ground_truth: torch.Tensor, metric="f1_score", split="val"
):
    """
    Finds the optimal threshold for binary classification based on a chosen metric.

    Args:
        logits (torch.Tensor): Tensor of shape (B, 1) containing model logits.
        ground_truth (torch.Tensor): Tensor of shape (B, 1) containing ground truth (0 or 1).
        metric (str): Metric to optimize ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cohen_kappa").
        split (str): Split name to ensure compatibility with calculate_metrics function.

    Returns:
        float: Optimal threshold.
        dict: Performance metrics at each threshold.
    """
    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits).view(-1).cpu().numpy()
    ground_truth = ground_truth.view(-1).cpu().numpy()

    # Initialize thresholds and metrics storage
    thresholds = np.arange(0, 1.01, 0.01)  # Range of thresholds from 0 to 1
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": [],
        "pr_auc": [],
        "cohen_kappa": [],
        "confusion_matrix": [],
        "threshold": [],
    }

    for threshold in thresholds:
        # Generate binary predictions at the current threshold
        predictions = (probabilities >= threshold).astype(int)

        # Convert predictions and ground_truth back to tensors for metrics calculation
        predictions_tensor = torch.tensor(predictions).view(-1, 1)
        ground_truth_tensor = torch.tensor(ground_truth).view(-1, 1)

        # Calculate metrics using the earlier function
        result = calculate_metrics_probs(
            probs=predictions_tensor,
            ground_truth=ground_truth_tensor,
            threshold=threshold,
        )

        # Extract the desired metric
        for k in result.keys():
            metrics[k].append(result[k])
        metrics["threshold"].append(threshold)

    # Find the threshold that maximizes the chosen metric
    metric_values = metrics[metric]
    optimal_idx = np.argmax(metric_values)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, metrics



# def ghost(
#     logits: torch.Tensor, ground_truth: torch.Tensor, n_splits = 4
# ) -> dict:
#     """
#     Calculate metrics for binary classification, including Cohen's Kappa.

#     Args:
#         logits (torch.Tensor): Tensor of shape (B, 1) containing model logits.
#         ground_truth (torch.Tensor): Tensor of shape (B, 1) containing ground truth (0 or 1).
#         threshold (float): Threshold for converting logits to binary predictions.
#         split (str): A string identifier for the data split (e.g., 'train', 'val', 'test').

#     Returns:
#         dict: Dictionary containing accuracy, precision, recall, F1 score, ROC AUC, PR AUC, confusion matrix, and Cohen's Kappa.
#     """

#     thresholds = np.arange(0.05, 0.95, 0.05)
#     metrics = {
#         "accuracy": [],
#         "precision": [],
#         "recall": [],
#         "f1_score": [],
#         "roc_auc": [],
#         "pr_auc": [],
#         "cohen_kappa": [],
#         "confusion_matrix": [],
#         "threshold": [],
#     }
#     table = {}
    
#     ground_truth = ground_truth.view(-1).cpu().numpy()
#     labels = np.array(ground_truth, dtype=int)
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
#     folds = []
    
#     for fold_idx, (_, subset_idx) in enumerate(skf.split(logits, labels), start=1):
#         subset_logits, subset_labels = logits[subset_idx], labels[subset_idx]
#         folds.append((subset_logits, subset_labels))

#     for threshold in thresholds: 
#         table[threshold] = []
        
#         for i in range(fold_idx): 
#             logits_fold = folds[i][0]
#             labels_fold = folds[i][1]
#             probs_fold = torch.sigmoid(logits_fold)
#             probs_fold = probs_fold.view(-1).cpu().numpy()
#             preds_fold = (probs_fold >= threshold).astype(int)
            

#         # Convert predictions and ground_truth back to tensors for metrics calculation
#             preds_tensor = torch.tensor(preds_fold).view(-1, 1)
#             labels_tensor = torch.tensor(labels_fold).view(-1, 1) 

#             # Calculate metrics using the earlier function
#             result = calculate_metrics_probs(
#                 probs=preds_tensor,
#                 ground_truth=labels_tensor,
#                 threshold=threshold,
#             )

#             table[threshold].append(result["f1_score"])

#     thr = {}

#     for threshold in thresholds: 
        
#         median_thr = torch.median(torch.tensor(table[threshold])).item()
#         thr[threshold] = median_thr
        
#     optimal_threshold = max(thr, key=thr.get)    

#     return optimal_threshold, metrics




def ghost(logits: torch.Tensor, ground_truth: torch.Tensor, n_splits=4) -> dict:
    """
    Calculate metrics for binary classification, including Cohen's Kappa.

    Args:
        logits (torch.Tensor): Tensor of shape (B, 1) containing model logits.
        ground_truth (torch.Tensor): Tensor of shape (B, 1) containing ground truth (0 or 1).
        n_splits (int): Number of splits for StratifiedKFold.

    Returns:
        dict: Dictionary containing optimal threshold and calculated metrics.
    """
    thresholds = np.arange(0.05, 0.95, 0.05)
    ground_truth = ground_truth.view(-1).cpu().numpy()
    labels = ground_truth.astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    folds = [
        (logits[idx].view(-1), labels[idx])
        for _, idx in skf.split(logits.cpu().numpy(), labels)
    ]

    metrics_table = {threshold: [] for threshold in thresholds}

    for threshold in thresholds:
        for logits_fold, labels_fold in folds:
            probs_fold = torch.sigmoid(logits_fold).cpu().numpy()
            preds_fold = (probs_fold >= threshold).astype(int)

            # Calculate metrics for current fold and threshold
            f1_score = calculate_f1_score(preds_fold, labels_fold)  # Assume this is defined elsewhere
            metrics_table[threshold].append(f1_score)

    # Determine optimal threshold based on median F1 score
    optimal_threshold = max(metrics_table, key=lambda t: np.median(metrics_table[t]))

    return optimal_threshold


def calculate_f1_score(preds, labels):
    """Calculate F1 score from predictions and labels."""
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

