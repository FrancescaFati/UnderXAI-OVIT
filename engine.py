"""
Engine module for PCS classification training and evaluation.

- Handles training, validation, and logging (optionally to wandb).
- Designed for public sharing: no user-specific or hardcoded paths.
- Optional dependencies (wandb, matplotlib, seaborn) are handled gracefully.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from torch.amp import GradScaler
from datetime import datetime
from rich import print
from rich.console import Console
from rich.theme import Theme
import metrics

# Optional dependencies
try:
    import wandb
except ImportError:
    wandb = None
    print("[yellow]wandb is not installed. Logging to wandb will be disabled.[/yellow]")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None
    print("[yellow]matplotlib or seaborn is not installed. Attention plotting will be disabled.[/yellow]")

custom_theme = Theme(
    {
        "epoch": "#d7af00",
        "batch": "cyan",
        "train": "cyan",
        "val": "dark_magenta",
        "val_log": "violet",
    }
)


def log_attention_weights(att_weights, step, prefix="Training", save=False, save_path=''):
    """
    Log attention matrix visualization to W&B (if available) and optionally save to disk.
    This function is robust to missing plotting/logging libraries.
    """
    if plt is None or sns is None:
        print("[yellow]matplotlib/seaborn not available: skipping attention plotting.[/yellow]")
        return
    if torch.is_tensor(att_weights):
        att_weights = att_weights.cpu().detach().numpy()
    num_samples = min(att_weights.shape[0], 4)
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    for sample_idx in range(num_samples):
        ax = axes[sample_idx] if num_samples > 1 else axes
        sns.heatmap(att_weights[sample_idx], 
                   ax=ax, 
                   cmap='viridis',
                   xticklabels=False, 
                   yticklabels=False)
        ax.set_title(f'Sample {sample_idx + 1}')
    plt.tight_layout()
    if wandb is not None:
        # Log the attention matrix as an image to wandb
        wandb.log({f"{prefix}/attention_matrix": wandb.Image(fig),"epoch":step},)
    if save: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {save_path}")
    plt.close()


class Engine:
    """
    Training and evaluation engine for PCS classification models.
    Handles training, validation, logging, and checkpointing.
    """
    def __init__(self, model, config, train_loader, val_loader, device):
        # Set up rich console for colored output
        self.console = Console(theme=custom_theme)
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Compute positive class weights for imbalanced data
        self.pos_weights = self._compute_pos_weight()
        # Select loss function based on config
        self.criterion = self.get_criterion()
        # AdamW optimizer with config parameters
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        # Scheduler for learning rate warmup and reduction
        self.warmup_steps = config.get("warmup_steps", 100)
        self.main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=config["scheduler_patience"],
            min_lr=1.e-8,
            threshold=1.e-4,
        )
        # Secondary scheduler (cosine or step) for additional LR scheduling
        schedulers = {
        'cosine': optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["epochs"]*len(train_loader)/config['scheduler_step'],  
            eta_min=0
        ),
        'step': optim.lr_scheduler.StepLR(
            self.optimizer, 
            config["epochs"]*len(train_loader)/config['scheduler_step'], 
            gamma=config['scheduler_gamma'])
        }
        self.secondary_scheduler = schedulers[config["scheduler_type"]]
        # For mixed precision training
        self.scaler = GradScaler()
        # Track best validation loss and F1 for early stopping
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.current_step = 0
        self.initial_lr = config["learning_rate"]
        self.num_classes = config["num_classes"]
        self.batch_size = config["batch_size"]
        self.clinical_features = config["clinical_features"]
        # For gradient accumulation
        self.gradient_accumulation_steps = config.get("accumulation_steps", 1)
        # Set up wandb logging if available
        if wandb is not None:
            wandb.watch(self.model,log_freq=2,log="all")
            wandb.define_metric("epoch")
            wandb.define_metric("train", step_metric="epoch")
            wandb.define_metric("val", step_metric="epoch")
            wandb.define_metric("Training", step_metric="epoch")
            wandb.define_metric("Validation_OPTHR", step_metric="epoch")
            wandb.define_metric("Validation", step_metric="epoch")
            wandb.define_metric("batch")
            wandb.define_metric("hyperp", step_metric="batch")
            wandb.define_metric("gradients", step_metric="batch")
            wandb.define_metric("parameters", step_metric="batch")

    def _compute_pos_weight(self):
        """
        Compute positive weight for BCEWithLogitsLoss from training data.
        This helps address class imbalance by weighting the positive class.
        """
        all_labels = []
        for batch in self.train_loader:
            labels = batch["label"]
            # Convert one-hot labels to binary (for binary classification)
            binary_labels = (labels[:, 1] > 0.5).cpu().numpy()
            all_labels.extend(binary_labels)
        labels_array = np.array(all_labels)
        class_counts = np.bincount(labels_array.astype(int), minlength=2)
        eps = 1e-8  # Avoid division by zero
        class_counts = np.maximum(class_counts, eps)
        pos_weight = class_counts[0] / class_counts[1]
        self.console.print(
            f"Class distribution: Negative={class_counts[0]}, Positive={class_counts[1]}", 
            style="epoch"
        )
        self.console.print(
            f"Computed positive weight: {pos_weight:.4f}", 
            style="epoch"
        )
        return torch.tensor([pos_weight], device=self.device)

    def get_criterion(self):
        """
        Return the loss function based on config.
        Supports standard BCE and weighted BCE for class imbalance.
        """
        if self.config["loss_function"] == "BCE":
            self.console.print(f"Using BCE loss", style="epoch")
            return nn.BCEWithLogitsLoss()
        elif self.config["loss_function"] == "WBCE":
            self.console.print(
                f"Using BCE loss with positive weight:[yellow]{self.pos_weights.item():.4f}[/yellow]",
                style="epoch",
            )
            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        raise ValueError(f"Unknown loss function: {self.config['loss_function']}")

    def format_tr_values(self,row, is_prediction=True):
        """
        Format values as TR 0/TR 1 with colors for rich console output.
        Used for displaying predictions and targets.
        """
        formatted = []
        for val in row:
            val = round(float(val), 1)
            tag = "TR 1" if val == 1.0 else "TR 0"
            color = "orange1" if val == 1.0 else "violet"
            formatted.append(f"[{color}]{tag:<12}[/{color}]")
        return formatted
    
    def format_truefalse_values(self,row):
        """
        Format values as True/False with colors for rich console output.
        Used for displaying accuracy results.
        """
        formatted = []
        for val in row:
            val = round(float(val), 1)
            tag = "True" if val == 1.0 else "False"
            color = "green" if val == 1.0 else "red"
            formatted.append(f"[{color}]{tag:<12}[/{color}]")
        return formatted
    
    def format_probs(self,probs):
        """Format probability pairs with consistent spacing"""
        if self.num_classes == 2:
            return [f"{round(float(pair[0]), 2):<6}, {round(float(pair[1]), 2):<6}" for pair in probs]
        elif self.num_classes == 1:
            return [f"{round(float(pair.item()), 2):<12}, " for pair in probs]

    
    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        self.console.print(
            f"Epoch [yellow]{epoch+1}/{num_epochs}[/yellow]", style="epoch"
        )
        self.console.print(f"TRAINING", style="train")

        metrics = {"Training/loss": 0.0}
        total_batches = len(self.train_loader)
        all_logits = []
        all_gts = []
        
        for batch_idx, batch in enumerate(self.train_loader):

            images = batch["image"].to(self.device)
            clinical_features = batch["clinical_features"].to(self.device)

            if self.num_classes == 1:
                labels = batch["label"][:, 1].to(self.device)
                labels = labels.unsqueeze(1)
            else:
                labels = batch["label"].to(self.device)


            # Forward pass and loss calculation
            logits, att_weights, _ = self.model(images, clinical_features)

            loss = self.criterion(logits, labels)
            scaled_loss = loss / self.gradient_accumulation_steps

            scaled_loss.backward()
            accumulated_loss = loss.item()

            # Gradient clipping and optimizer step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if att_weights is not None:
                    log_attention_weights(att_weights, step=epoch, prefix="Training")
                wandb.log({"LR":self.optimizer.param_groups[0]['lr']})
                
            # Update metrics
            all_logits.append(logits.detach())
            all_gts.append(labels.detach())
            
            with torch.no_grad():
                
                if self.num_classes == 1:
                    probs = torch.sigmoid(logits)
                    binary_preds = (probs > 0.5).float()
                    binary_targets = labels
                else:
                    probs = logits
                    binary_preds = logits.argmax(dim=1)
                    binary_preds = binary_preds.unsqueeze(1)
                    binary_targets = labels.argmax(dim=1)
                    binary_targets = binary_targets.unsqueeze(1)

                metrics["Training/loss"] += accumulated_loss

                
            self.secondary_scheduler.step()
            # Log progress
            self.console.print(
                f"Batch [cyan]{batch_idx+1}/{total_batches}[/cyan]", style="batch"
            )   

            # First, format all the data
            prob_strs = [self.format_probs(probs)]
            pred_strs = [self.format_tr_values(row) for row in binary_preds]
            targ_strs = [self.format_tr_values(row) for row in binary_targets]
            acc_strs = [self.format_truefalse_values(row) for row in (binary_preds == binary_targets).tolist()]

            # Then print everything aligned
            self.console.print(
                f">> Probs:[white]{prob_strs}[/white]",
                style="bright_cyan",
            )
            self.console.print(
                f">> Preds:[white]{pred_strs}[/white]",
                style="bright_cyan",
            )
            self.console.print(
                f">> Targs:[white]{targ_strs}[/white]",
                style="bright_cyan",
            )
            self.console.print(
                f">> Acc  :{acc_strs}",
                style="bright_cyan",
            )
            self.console.print(
                f">> Loss :[white]{accumulated_loss:.3f}[/white]", style="bright_cyan"
            )
            self.console.print()

            self.log_batch_metrics(batch_idx)   
            
                
            del(images, labels, logits, loss)
        
        return {k: v / total_batches for k, v in metrics.items()}, att_weights, torch.cat(all_logits, dim=0), torch.cat(all_gts, dim=0)
    

    def validate(self, epoch, num_epochs):
        self.model.eval()
        self.console.print(
            f"Epoch [yellow]{epoch+1}/{num_epochs}[/yellow]", style="epoch"
        )
        self.console.print(f"VALIDATION", style="val")

        metrics = {"Validation/loss": 0.0}
        total_batches = len(self.val_loader)
        all_logits = []
        all_gts = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch["image"].to(self.device)
                clinical_features = batch["clinical_features"].to(self.device)

                if self.num_classes == 1:
                    labels = batch["label"][:, 1].to(self.device)
                    labels = labels.unsqueeze(1)
                else:
                    labels = batch["label"].to(self.device)


                logits,att_weights,_ = self.model(images, clinical_features)

                loss = self.criterion(logits, labels)
                
                if att_weights is not None:
                    log_attention_weights(att_weights, step=epoch, prefix="Validation")

                if self.num_classes == 1:
                    probs = torch.sigmoid(logits)
                    binary_preds = (probs > 0.5).float()
                    binary_targets = labels
                else:
                    probs = logits
                    binary_preds = logits.argmax(dim=1)
                    binary_preds = binary_preds.unsqueeze(1)
                    binary_targets = labels.argmax(dim=1)
                    binary_targets = binary_targets.unsqueeze(1)

                all_logits.append(logits.detach())
                all_gts.append(labels.detach())
                
                metrics["Validation/loss"] += loss.item()

                # Log progress
                self.console.print(
                    f"Batch [magenta]{batch_idx+1}/{total_batches}[/magenta]",
                    style="val_log",
                )    
                                
                prob_strs = [self.format_probs(probs)]
                pred_strs = [self.format_tr_values(row) for row in binary_preds]
                targ_strs = [self.format_tr_values(row) for row in binary_targets]
                acc_strs = [self.format_truefalse_values(row) for row in (binary_preds == binary_targets).tolist()]

                # Then print everything aligned
                self.console.print(
                    f">> Probs:[white]{prob_strs}[/white]",
                    style="magenta",
                )
                self.console.print(
                    f">> Preds:[white]{pred_strs}[/white]",
                    style="magenta",
                )
                self.console.print(
                    f">> Targs:[white]{targ_strs}[/white]",
                    style="magenta",
                )
                self.console.print(
                    f">> Acc  :{acc_strs}",
                    style="magenta",
                )
                self.console.print(
                    f">> Loss :[white]{loss:.3f}[/white]", style="magenta"
                )
                self.console.print()

                del(images, labels, logits, loss)
        return {k: v / total_batches for k, v in metrics.items()}, torch.cat(all_logits, dim=0), torch.cat(all_gts, dim=0)



    def update_scheduler(self, val_loss):
        """Basic scheduler update without warmup"""
        if self.current_step < self.warmup_steps and self.warmup_steps > 0:
            old_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = self.optimizer.param_groups[0]["lr"]
            self.current_step += 1
        else:
            old_lr = self.optimizer.param_groups[0]["lr"]
            self.main_scheduler.step(val_loss)

            new_lr = self.optimizer.param_groups[0]["lr"]

        if new_lr != old_lr:
            print(f"[red]Learning rate changed: {old_lr:.2e} -> {new_lr:.2e}[/red]")
            print()
        else:
            print(f"[green]Current Learning rate: {new_lr:.2e}[/green]")
            print()

    def train(self, num_epochs: int, resume_from: str = None):
        start_epoch = 0
        best_model_state = None
        no_improvement_epochs = 0
        best_val_loss = float("inf")
        prev_train_metrics = {"Training/loss": 0.0}
        prev_val_metrics = {"Validation/loss": 0.0}


        if resume_from:
            start_epoch, _ = self.load_checkpoint(resume_from)
            print(f"Resuming from epoch {start_epoch}")

        self.console.print(
            f"Initial learning rate: [yellow]{self.initial_lr}[/yellow]", style="epoch"
        )
        self.console.print(
            f"Batch size: [yellow]{self.batch_size}[/yellow]", style="epoch"
        )
        self.console.print(
            f"Gradient accumulation steps: [yellow]{self.gradient_accumulation_steps}[/yellow]",
            style="epoch",
        )

        trainable_tensors = sum(p.requires_grad for p in self.model.parameters())
        total_tensors = sum(1 for p in self.model.parameters())
        self.console.print(
            f"Trainable parameters:[yellow]{trainable_tensors}/{total_tensors}[/yellow]",
            style="epoch",
        )
        print()

        if trainable_tensors == 0:
            raise ValueError("No trainable parameters found in the model!")

        for epoch in range(start_epoch, num_epochs):

            # Training phase 
            train_metrics, att_weights, logits, gts = self.train_epoch(epoch, num_epochs)
            comprehensive_train_metrics = metrics.calculate_metrics(logits, gts, 0.5, split="Training")

            comprehensive_train_metrics.update({"epoch":epoch})
            wandb.log(train_metrics)
            wandb.log(comprehensive_train_metrics)
            
            val_metrics,logits, gts = self.validate(epoch, num_epochs)
            comprehensive_val_metrics = metrics.calculate_metrics(logits, gts, 0.5, split="Validation")

            comprehensive_val_metrics.update({"epoch":epoch})
            wandb.log(val_metrics)
            wandb.log(comprehensive_val_metrics)



            self.log_epoch_metrics(epoch, train_metrics, val_metrics)

            self.console.print(
                f"SUMMARY EPOCH [yellow]{epoch+1}/{num_epochs}[/yellow]", style="epoch"
            )
            self.console.print(
                f"TRAIN[cyan] Loss:{prev_train_metrics.get('Training/loss', -1):.4f} -> {train_metrics.get('Training/loss',-1):.4f}[/cyan]",
                style="batch",
            )
            self.console.print(
                f"VAL  [magenta] Loss:{prev_val_metrics.get('Validation/loss',-1):.4f} -> {val_metrics.get('Validation/loss',-1):.4f}[/magenta]",
                style="val_log",
            )
            print()
    
                  
            prev_train_metrics = train_metrics
            prev_val_metrics = val_metrics

            # Update learning rate based on validation loss
            self.update_scheduler(val_metrics["Validation/loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Save best model based on validation loss
            if val_metrics["Validation/loss"] < best_val_loss:

                best_val_loss = val_metrics["Validation/loss"]
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr": current_lr,
                }
                no_improvement_epochs = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                no_improvement_epochs += 1

            # Track best F1 score

            if comprehensive_val_metrics["Validation/f1_score"] > self.best_val_f1:
                self.best_val_f1 = comprehensive_val_metrics["Validation/f1_score"]


            # Early stopping check
            if no_improvement_epochs >= self.config["early_stopping_patience"]:
                self.console.print(
                    f"STOP TRAINING: early stopping triggered after [red]{epoch + 1}[/red] epochs",
                    style="red",
                )
                self.console.print(
                    f"Best validation loss: [red]{best_val_loss:.4f}[/red]", style="red"
                )

                self.console.print(
                    f"Best validation F1: [red]{self.best_val_f1:.4f}[/red]", style="red"
                )

                if best_model_state is not None:
                    self.console.print(
                        f"Restoring best model from epoch [red]{best_model_state['epoch']}[/red]",
                        style="red",
                    )
                    self.restore_best_model(best_model_state,att_weights,epoch)
                break
            

            
        if epoch == num_epochs - 1:
            self.console.print(f"Training completed. Restoring best model from epoch [yellow]{best_model_state['epoch']}[/yellow]", style="epoch")
            self.restore_best_model(best_model_state,att_weights,epoch)



    def restore_best_model(self, best_model_state, att_weigths, epoch):
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state["model_state_dict"])
            self.optimizer.load_state_dict(best_model_state["optimizer_state_dict"])
            log_attention_weights(att_weigths, step = epoch, save = False, save_path=f"attention_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{epoch+1}.png")


    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "best_val_loss": self.best_val_loss,
                "best_val_f1": self.best_val_f1,
                "current_step": self.current_step,
                "current_lr": self.optimizer.param_groups[0]["lr"],
                "config": {k: v for k, v in self.config.items()},
            }

            save_path = f"best_model_{wandb.run.id}.pt"
            torch.save(checkpoint, save_path)
            self.console.print(f"Saved checkpoint", style="epoch")

        except Exception as e:
            self.console.print(
                f"Warning: Failed to save checkpoint: {str(e)}", style="red"
            )
            try:
                fallback_path = f"model_only_epoch_{epoch}.pt"
                torch.save(self.model.state_dict(), fallback_path)
                self.console.print(
                    f"Saved model-only fallback to {fallback_path}", style="red"
                )
            except Exception as e2:
                self.console.print(
                    f"Critical: Failed to save even model-only checkpoint: {str(e2)}",
                    style="red",
                )

    def log_batch_metrics(self, batch_idx: int, metrics: Dict[str, float]=None):
        """Log batch-level metrics to wandb"""
        wandb.log(
            {
                "batch": batch_idx,
                "batch/learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        )

    def log_epoch_metrics(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ):
        """Log epoch-level metrics to wandb"""
        metrics = {
            "epoch": epoch,
            "hyperp/learning_rate": self.optimizer.param_groups[0]["lr"],
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        wandb.log(metrics)

