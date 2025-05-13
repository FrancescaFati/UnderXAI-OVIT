#%%
import wandb
import torch 
from rich import print
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from typing import Tuple, List
import numpy as np
from torchmetrics.functional.classification import binary_f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import os
import subprocess
import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/home/ffati/MedViT')
from MedViT import MedViT_small

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler()

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature (float): Temperature scaling factor.
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Computes the supervised contrastive loss.
        
        Args:
            features (torch.Tensor): Tensor of shape [batch_size, feature_dim] containing the features.
            labels (torch.Tensor): Tensor of shape [batch_size] with class labels (e.g. 1 for tumor, 0 for non-tumor).

        Returns:
            torch.Tensor: The computed loss.
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize feature vectors to unit norm
        features = F.normalize(features, dim=1)
        
        # Compute cosine similarity between all pairs: [batch_size, batch_size]
        logits = torch.matmul(features, features.T) / self.temperature

        # Create mask: 1 if samples share the same label, 0 otherwise.
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-comparisons by zeroing out the diagonal
        logits_mask = 1 - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # For numerical stability, subtract the maximum logit for each anchor.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean log-likelihood over positive pairs for each sample
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss is the negative average of these log-likelihoods.
        loss = -mean_log_prob_pos.mean()
        return loss


# Surrogate Fβ Loss class
class SurrogateFbetaLoss(torch.nn.Module):
    def __init__(self, beta, pos_proportion, num_classes):
        super(SurrogateFbetaLoss, self).__init__()
        self.beta = beta
        self.pos_proportion = pos_proportion  # p = proportion of positives
        self.num_classes = num_classes
        

    def forward(self, logits, targets):
        if self.num_classes == 1: 
            probs = torch.sigmoid(logits)
        else: 
            probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
            
        pos_loss = -targets * torch.log(probs + 1e-8)  # Positive term
        neg_loss = (1 - targets) * torch.log(
            self.beta**2 * self.pos_proportion / (1 - self.pos_proportion) + probs + 1e-8
        )  # Negative term
        return (pos_loss + neg_loss).mean()
    
def train(model: Module, vit_transform, loader: DataLoader, optimizer: Optimizer, loss_function: Module, epoch, accum_step) -> Tuple[float, float, float]:
    """
    Trains the model for one epoch over the given DataLoader.

    Args:
        model (Module): The neural network model to be trained.
        loader (DataLoader): DataLoader containing training data.
        optimizer (Optimizer): Optimizer used for training.
        loss_function (Module): Loss function used for training.

    Returns:
        Tuple[float, float, float]: Returns a tuple containing the average loss, average accuracy, and F1 score for the epoch.
    """
    
    model.train()
    cumu_loss = 0
    accuracy = 0
    f1score = 0

    all_targets = []
    all_predictions = []

    optimizer.zero_grad()


    for b, batch in enumerate(loader):
        print(f"[orange3]TRAIN {b}/{len(loader)}[/orange3] ")
        
        images, target = batch['image'].to(device), batch['label'].to(device)
        #optimizer.zero_grad()

        # with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
        #     prediction = model(images)
        #     loss = loss_function(prediction, target)
        #prediction = model(images, vit_transform)
        prediction = model(images)     
        #print("Logits:\n", prediction.permute(1, 0).cpu().detach().numpy())
        #print("Probabilities: \n", torch.sigmoid(prediction).permute(1,0).cpu().detach().numpy())
    
        
        # ➡ Forward pass
        loss = loss_function(prediction, target)
        loss = loss / accum_step

        # ⬅ Backward pass + weight update
        loss.backward()
        
        if (b + 1) % accum_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        cumu_loss += loss.item()
        wandb.log({"train/train_loss": loss.item()})

        value = torch.eq(prediction.argmax(dim=1), target.argmax(dim=1))
        accuracy += value.sum().item() / len(images)

        f1score_batch = binary_f1_score(prediction, target)
        # wandb.log({"train/train_f1_score":f1score_batch.item()})

        all_targets.extend(target.argmax(dim=1).cpu().numpy())
        all_predictions.extend(prediction.argmax(dim=1).cpu().numpy())
        # total_norm = 0
        # for p in model.parameters():
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm**0.5
        
        # print(
        #     f">>> Train batch loss: {loss.item()} - Gradient Norm: {total_norm} - \nTARG: {target.argmax(dim=1).cpu().detach().numpy()} - \nPRED: {prediction.argmax(dim=1).cpu().detach().numpy()}"
        # )
        
        print(
                f">>> Train batch loss: {loss.item()} - \nTARG: {target.argmax(dim=1).cpu().detach().numpy()} - \nPRED: {prediction.argmax(dim=1).cpu().detach().numpy()}- \nLOGI: {prediction.cpu().detach().numpy()}"
        )

        wandb.log(
            {
                "TRAIN Confusion Matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=target.argmax(dim=1).cpu().numpy(),
                    preds=prediction.argmax(dim=1).cpu().numpy(),
                    class_names=[0, 1],
                )
            }
        )

        # TODO: ROC

        #wandb.log({"TRAIN ROC" : wandb.plot.roc_curve(target.argmax(dim=1).cpu().numpy(), prediction.argmax(dim=1).cpu().numpy(), labels=[0, 1, 2, 3], classes_to_plot=None)})
        # wandb.log({"Train pr":wandb.plot.pr_curve(target.argmax(dim=1).cpu().numpy(), prediction.argmax(dim=1).cpu().numpy(), labels=[0, 1], classes_to_plot=None)})

        del(images, target)
        # hook_handle.remove()
        
        
    
    f1score_epoch = binary_f1_score(
        torch.tensor(np.array(all_predictions)), torch.tensor(np.array(all_targets))
    )
    return cumu_loss / (b + 1), accuracy / (b + 1), f1score_epoch

        

def val(model: Module, vit_transform, loader: DataLoader, loss_function: Module) -> Tuple[float, float, float]:
    """
    Validates the model over the given DataLoader.

    Args:
        model (Module): The neural network model to be validated.
        loader (DataLoader): DataLoader containing validation data.
        loss_function (Module): Loss function used for validation.

    Returns:
        Tuple[float, float, float]: Returns a tuple containing the average loss, average accuracy, and F1 score for the validation set.
    """

    cumu_loss = 0.0
    accuracy = 0.0
    f1score = 0.0
    num_tot = 0.0

    all_targets = []
    all_predictions = []

    model.eval()
    for b, batch in enumerate(loader):
        print(f"[green]VAL {b}/{len(loader)}[/green] ")

        images, target = batch['image'].to(device), batch['label'].to(device)

        with torch.no_grad():

            prediction = model(images, vit_transform)
            #print("Logits: \n", prediction.permute(1, 0).cpu().detach().numpy())
            #print("Probabilities: \n", torch.sigmoid(prediction).permute(1,0).cpu().detach().numpy())

            loss = loss_function(prediction, target)
            cumu_loss += loss.item()

            # scheduler.step(loss)

            value = torch.eq(prediction.argmax(dim=1), target.argmax(dim=1))
            accuracy += value.sum().item() / len(images)

            f1score_batch = binary_f1_score(prediction, target)
            # wandb.log({"val/val_f1_score":f1score_batch.item()})

            all_targets.extend(target.argmax(dim=1).cpu().numpy())
            all_predictions.extend(prediction.argmax(dim=1).cpu().numpy())
            

            
            print(
                f">>> Val batch loss: {loss.item()} - \nTARG: {target.argmax(dim=1).cpu().detach().numpy()} - \nPRED: {prediction.argmax(dim=1).cpu().detach().numpy()}- \nLOGI: {prediction.cpu().detach().numpy()}"
            )

            # Log one batch of images to the dashboard, always same batch_idx.
            # if i==batch_idx and log_images:
        # log_image_table(prediction, target)

                
        wandb.log(
            {
                "VAL Confusion Matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=target.argmax(dim=1).cpu().numpy(),
                    preds=prediction.argmax(dim=1).cpu().numpy(),
                    class_names=[0, 1],
                )
            }
        )

        # TODO: ROC
        # ROC

        #wandb.log({"VAL ROC" : wandb.plot.roc_curve(target.argmax(dim=1).cpu().numpy(), prediction.argmax(dim=1).cpu().numpy(), labels=[0, 1, 2, 3], classes_to_plot=None)})
        # wandb.log({"Val pr":wandb.plot.pr_curve(target.argmax(dim=1).cpu().numpy(), prediction.argmax(dim=1).cpu().numpy(), labels=[0, 1], classes_to_plot=None)})

        # wandb.log({'roc': wandb.plots.ROC(target.argmax(dim=1).cpu().numpy(), prediction.argmax(dim=1).cpu().numpy(), [0,1])})
        # TODO: add nice visualization in wandb --> https://wandb.ai/site/solutions/classification-regression

        del (images, target)

    f1score_epoch = binary_f1_score(
        torch.tensor(np.array(all_predictions)), torch.tensor(np.array(all_targets))
    )

    return cumu_loss / (b + 1), accuracy / (b + 1), f1score_epoch


def windowing(image: np.ndarray, window_level: int = 40, window_width: int = 400) -> np.ndarray:
    """
    Applies windowing to a medical image, enhancing contrast within a specific intensity range.

    Parameters:
    - image (np.ndarray): The input image.
    - window_level (int): The center of the intensity window.
    - window_width (int): The width of the intensity window.

    Returns:
    - np.ndarray: The windowed image.
    """
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - lower_bound) / window_width
    return image



def open_txt_file(txt_file: str) -> Tuple[List[str], List[int]]:
    """
    Reads a text file where each line is a file path. Extracts target values based on the
    directory structure of the paths and returns two lists: one with the image paths and
    another with the corresponding target values.

    The target value is assumed to be a directory name in the path, specifically the
    third directory from the end of the path.

    Example of a line in the text file:
    /home/ffati/Under-XAI/dataset/train/0/CC03014885/CT_crop.nii.gz
    Here, '0' is extracted as the target value.

    Args:
        txt_file (str): The path to the text file containing the image file paths.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing two lists:
                                      - The first list contains the image paths as strings.
                                      - The second list contains the corresponding target values as integers.
    """
    targets = []
    im_paths = []

    with open(txt_file, 'r') as file:
        for line in file:
            # Remove newline characters and any trailing spaces
            line = line.strip()
            
            # Assuming the target value is the second directory in the path
            # Example: /home/ffati/Under-XAI/dataset/train/0/CC03014885/CT_crop.nii.gz
            # Split the path and extract the target value
            parts = line.split('/')
            target = parts[-3]  # This should adjust based on your path's structure

            # Append to lists
            im_paths.append(line)
            targets.append(int(target))

    
    return im_paths, targets

def open_txt_file_specify(txt_file: str) -> Tuple[List[str], List[int]]:
    """
    Reads a text file where each line is a file path. Extracts target values based on the
    directory structure of the paths and returns two lists: one with the image paths and
    another with the corresponding target values.

    The target value is assumed to be a directory name in the path, specifically the
    third directory from the end of the path.

    Example of a line in the text file:
    /home/ffati/Under-XAI/dataset/train/0/CC03014885/CT_crop.nii.gz
    Here, '0' is extracted as the target value.

    Args:
        txt_file (str): The path to the text file containing the image file paths.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing two lists:
                                      - The first list contains the image paths as strings.
                                      - The second list contains the corresponding target values as integers.
    """


    with open(txt_file, 'r') as file:
        lines = file.readlines()
        paths = []
        class_labels = []
        
        for line in lines:
            path, class_label = line.strip().split(',')
            paths.append((path))
            class_labels.append(int(class_label))
    
    return paths, class_labels

from typing import List, Tuple


def open_txt_file_group(txt_file: str) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Reads a text file where each line contains a file path and a class label separated by a comma.
    Groups and returns paths and class labels by class label values (0 to 3), ordered by class.

    Args:
        txt_file (str): The path to the text file containing the image file paths and class labels.

    Returns:
        Tuple[List[List[str]], List[List[int]]]: A tuple containing two lists of lists:
                                      - The first list contains lists of image paths for each class label (0 to 3).
                                      - The second list contains lists of corresponding target values for each class label.
    """
    
    paths = []
    class_labels = []
    
    # Read lines from file and collect paths and class labels
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            path, class_label = line.strip().split(',')
            paths.append(path)
            class_labels.append(int(class_label))
    
    # Combine paths and labels into pairs and sort by label
    data = sorted(zip(paths, class_labels), key=lambda x: x[1])
    
    # Separate data into groups by class label
    paths_grouped = [[] for _ in range(4)]
    class_labels_grouped = [[] for _ in range(4)]
    
    for path, class_label in data:
        paths_grouped[class_label].append(path)
        class_labels_grouped[class_label].append(class_label)
    
    return paths_grouped, class_labels_grouped


def read_txt_file(txt_file: str):

    im_paths = []

    with open(txt_file, 'r') as file:
        for line in file:
            # Remove newline characters and any trailing spaces
            line = line.strip()
            im_paths.append(line)

    return im_paths


class LabeledDataset(Dataset):
    def __init__(self, paths):
        self.data = []
        self.labels = []
        for path in paths:
            data = torch.load(path)
            label = int(Path(path).parts[-2])
            one_hot_label = F.one_hot(torch.tensor(label), num_classes=2).to(torch.float)
            self.data.append(data)
            self.labels.append(one_hot_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'image': self.data[idx],
            'label': self.labels[idx]
        }

#------------------------------------------------------------------------------------#
def log_feature_maps(feature_maps, layer_name, epoch, batch_idx):
    
    feature_map = feature_maps[0].detach().cpu()  # Take the first feature map tensor
    
    if feature_map.dim() == 5:  # Check if it's a batch of 3D data: (B, C, W, H, D)
        feature_map = feature_map[0]  # Take the features of the first item in the batch
        
     # Select a middle slice to visualize, for simplicity
    slice_idx = feature_map.shape[-1] // 2
    fig, ax = plt.subplots(figsize=(50, 50))
    
    for i in range(min(feature_map.shape[0], 4)):  # Show up to 4 feature maps
        ax.clear()
        ax.imshow(feature_map[i, :,:, slice_idx].numpy(), cmap='gray')
        ax.axis('off')
        # Log to wandb
        wandb.log({f"{layer_name}_Epoch_{epoch}_Batch_{batch_idx}_Map_{i}": [wandb.Image(plt)]})
        
    plt.close(fig)
#------------------------------------------------------------------------------------#

# def log_feature_maps(output_features, epoch, batch_idx):
#     for layer_name, feature_maps in output_features.items():
#         # Iterate through all feature maps collected for this layer
#         for feature_map_idx, feature_map in enumerate(feature_maps):
#             feature_map = feature_map.detach().cpu()  # Assuming (B, C, D, H, W)
#             b, c, w, h, d = feature_map.shape
#             # Select a representative slice, here choosing the middle one for simplicity
#             slice_idx = d // 2
            
#             fig, axes = plt.subplots(1, min(c, 4), figsize=(20, 5))  # Adjust subplot size as needed
#             if c < 4:
#                 axes = [axes]  # Ensure axes is iterable for single channel
#             for i, ax in enumerate(axes):
#                 if i >= c:  # Safety check to avoid indexing errors
#                     break
#                 ax.imshow(feature_map[0, i,:,:,slice_idx].numpy(), cmap='gray')  # Showing the first batch's middle slice
#                 ax.axis('off')
#                 ax.set_title(f"Channel {i+1}")
#             plt.tight_layout()
#             plt.suptitle(f"{layer_name} - Epoch {epoch} - Batch {batch_idx} - Map {feature_map_idx}", y=1.05)
            
#             # Log to wandb, adjust the log name to include layer_name and feature_map_idx
#             wandb.log({f"{layer_name}_Epoch_{epoch}_Batch_{batch_idx}_Feature_{feature_map_idx}": [wandb.Image(plt)]})
#             plt.close(fig)
            
#---------------------------------------------------------------------------------------------------------#

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[red] EARLY STOPPING COUNTER: {self.counter} / {self.patience} [red]')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Save the model when validation loss decrease.'''
        if self.verbose:
            print(f'[orange]VAL LOSS: ({self.val_loss_min:.6f} --> {val_loss:.6f})[orange]')
        self.val_loss_min = val_loss
        # Puoi salvare il modello qui, ad esempio:
        # torch.save(model.state_dict(), 'checkpoint.pt')

def detect_aval_cpus():
    """
    Detects the number of available CPUs.
    """
    try:
        currentjobid = os.environ["SLURM_JOB_ID"]
        currentjobid = int(currentjobid)
        command = f"squeue --Format=JobID,cpus-per-task | grep {currentjobid}"
        # Run the command as a subprocess and capture the output
        output = subprocess.check_output(command, shell=True)[5:-4].replace(b" ", b"")
        cpus = output.decode("utf-8")
        cpus2 = len(os.sched_getaffinity(0))
        cpus = min(int(cpus), cpus2)
    except:
        cpus = 1 #os.cpu_count()
    return cpus


class MedViT_FeatureExtractor(nn.Module):
    def __init__(self):
        super(MedViT_FeatureExtractor, self).__init__()
        
        medvit = MedViT_small(pretrained=True)
        self.stem = medvit.stem
        self.features = medvit.features
        self.norm = medvit.norm
        self.avgpool = medvit.avgpool  # Keep avgpool for global feature extraction

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.norm(x)
        x = self.avgpool(x)  # Output shape: (B, 1024, 1, 1)
        x = x.flatten(1)     # Flatten to (B, 1024)
        return x
    
    def num_features(self):
        """Return the number of output features."""
        return self.avgpool.shape[1]
    
    
def print_params_analysis(model):
    print("\n[yellow]Parameter Analysis[/yellow]")
    total_params = 0
    total_tensors = 0
    trainable_params = 0
    trainable_tensors = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()  # Number of individual parameters
        is_trainable = param.requires_grad
        
        # print(f"\n{name}:")
        # print(f"  Shape: {param.shape}")
        # print(f"  Parameters: {num_params:,}")
        # print(f"  Trainable: {is_trainable}")
        
        total_params += num_params
        total_tensors += 1
        if is_trainable:
            trainable_params += num_params
            trainable_tensors += 1
    
    print(f"[yellow]Total parameter tensors[/yellow]: [yellow]{total_tensors}[/yellow]")
    print(f"[yellow]Trainable parameter tensors[/yellow]: [yellow]{trainable_tensors}/{total_tensors}[/yellow]")
    print(f"[yellow]Total individual parameters[/yellow]: [yellow]{total_params:,}[/yellow]")
    print(f"[yellow]Trainable individual parameters[/yellow]: [yellow]{trainable_params:,}[/yellow]")


class SurrogateFBetaLoss(nn.Module):
    def __init__(self, beta=1.0, p=None, epsilon=1e-7):
        """
        Surrogate F-beta loss function for binary classification with imbalanced data.
        
        Args:
            beta (float): Beta parameter that controls precision-recall trade-off
                         beta > 1 favors recall over precision
                         beta < 1 favors precision over recall
            p (float): Proportion of positive samples in the dataset.
                      If None, will be estimated from the target labels.
            epsilon (float): Small constant to prevent numerical instability
        """
        super().__init__()
        self.beta = beta
        self.p = p
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        """
        Compute the surrogate F-beta loss.
        
        Args:
            y_pred (torch.Tensor): Predicted probabilities (after SOFTMAX)
            y_true (torch.Tensor): Ground truth binary labels
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Ensure input is valid
        if y_pred.shape != y_true.shape:
            raise ValueError("Predictions and targets must have the same shape")
                        
        # Compute positive and negative parts of the loss
        pos_loss = -y_true * torch.log(y_pred + self.epsilon)
        
        beta_squared = self.beta * self.beta
        p_term = beta_squared * self.p / (1 - self.p)
        neg_loss = (1 - y_true) * torch.log(p_term + y_pred + self.epsilon)
        
        # Combine losses
        loss = pos_loss + neg_loss
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', epsilon=1e-6):
        """
        Initialize Focal Loss with improved parameter handling and numerical stability.
        
        Args:
            alpha (float or torch.Tensor): Weighting factor for class balance.
                                         If float: global weighting
                                         If tensor: class-wise weights
                                         If None: all classes weighted equally
            gamma (float): Focusing parameter for modulating loss (must be non-negative)
            reduction (str): 'mean', 'sum' or 'none' for the reduction type
            epsilon (float): Small constant for numerical stability
        """
        super(FocalLoss, self).__init__()
        
        # Validate gamma
        if not isinstance(gamma, (float, int)) or gamma < 0:
            raise ValueError(f"Gamma should be a non-negative float or integer, got: {gamma}")
        self.gamma = gamma
        
        # Validate and process alpha
        self.alpha = self._validate_alpha(alpha)
        
        # Validate reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction should be 'mean', 'sum' or 'none', got: {reduction}")
        self.reduction = reduction
        
        # Set epsilon for numerical stability
        self.epsilon = epsilon

    def _validate_alpha(self, alpha):
        """
        Validate and process the alpha parameter.
        
        Args:
            alpha: Input alpha parameter (None, float, or tensor)
            
        Returns:
            torch.Tensor or None: Processed alpha weights
        """
        if alpha is None:
            return None
            
        if isinstance(alpha, (float, int)):
            if not 0 <= alpha <= 1:
                raise ValueError(f"Alpha should be between 0 and 1, got: {alpha}")
            return torch.tensor([alpha, 1-alpha])
            
        if torch.is_tensor(alpha):
            if alpha.dim() != 1:
                raise ValueError("Alpha tensor should be 1D")
            if not torch.all((alpha >= 0) & (alpha <= 1)):
                raise ValueError("All alpha values should be between 0 and 1")
            return alpha.to(torch.float32)  # Ensure float32 type
            
        raise ValueError(f"Alpha should be None, float, or tensor, got: {type(alpha)}")

    def forward(self, inputs, targets):
        """
        Calculate the focal loss with improved numerical stability.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Ensure inputs and targets are float32
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)
        
        # Validate inputs and targets
        if inputs.shape != targets.shape:
            raise ValueError(f"Inputs and targets should have the same shape. Got {inputs.shape} and {targets.shape}")
        
        # Calculate binary cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate probabilities
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, 
                                self.alpha[1].to(inputs.device), 
                                self.alpha[0].to(inputs.device))
            ce_loss = alpha_t * ce_loss

        # Calculate focal term with numerical stability
        focal_term = ((1 - pt) + self.epsilon) ** self.gamma
        
        # Calculate final focal loss
        focal_loss = focal_term * ce_loss

        # Apply reduction if specified
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss