"""
Main training and evaluation script for PCS classification.

- Accepts a config file (JSON) as a command-line argument.
- No hardcoded credentials or user-specific paths.
- Designed for sharing in a public repository.
"""

import argparse
import json
import wandb
import torch
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from pcs_dataset import PCSDataset
from engine import Engine
from model import Vit_Classifier, Slice_Contrastive, Volume_Contrastive
from eval import ModelEvaluator
from utilis import detect_aval_cpus
from rich import print
import numpy as np
import pandas as pd

def sweep(config=None):
    """
    Run a training and evaluation sweep with the given config dictionary.
    Handles model instantiation, data loading, training, saving, and evaluation.
    Args:
        config (dict): Configuration dictionary loaded from JSON.
    """
    with wandb.init(
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Instantiate model
        model = Vit_Classifier(config).to(device)
        print("Model loaded")

        # Prepare datasets for training, validation, and testing
        train_dataset = PCSDataset(
            data_file=config['train_paths'],
            excluded_years=[2015],
            transform=None,
            is_training=True,
            seg=False,
            seg_as_im=True,
            clinical_fe=True,
        )
        val_dataset = PCSDataset(
            data_file=config['val_paths'],
            excluded_years=[2015],
            transform=None,
            is_training=False,
            seg=False,
            seg_as_im=True,
            clinical_fe=True,
        )
        test_dataset = PCSDataset(
            data_file=config['test_paths'],
            excluded_years=[2015],
            transform=None,
            is_training=False,
            seg=False,
            seg_as_im=True,
            clinical_fe=True,
        )

        # Create data loaders for each dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=detect_aval_cpus(),
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=detect_aval_cpus(),
            pin_memory=True,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=detect_aval_cpus(),
            pin_memory=True,
            drop_last=False
        )

        # Initialize the training engine
        engine = Engine(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        # Train the model
        engine.train(num_epochs=config['epochs'])

        # Create directory for saving models
        save_dir = Path('models').absolute()
        print(f"[yellow]Creating directory: {save_dir}[/yellow]")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save only essential data that can be pickled
        save_dict = {
            'model_state_dict': model.state_dict(),
            'learning_rate': config.get('learning_rate'),
            'batch_size': config.get('batch_size'),
            'epochs': config.get('epochs'),
        }
        save_path = save_dir / f"final_model_{wandb.run.id}.pt"
        torch.save(save_dict, str(save_path))
        print(f"[yellow]Saving model to: {save_path}[/yellow]")

        # Evaluate the model on the test set and save metrics
        print()
        print("[orange3]TEST[/orange3]")
        evaluator = ModelEvaluator(model, device, config=config)
        comprehensive_test_metrics = evaluator.predict(test_loader)
        df_test_metrics = pd.DataFrame(comprehensive_test_metrics)
        df_test_metrics.to_csv(f"test_metrics_{wandb.run.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

def load_config(config_path):
    """
    Load a JSON config file from the given path and validate required fields.
    Args:
        config_path (str): Path to the JSON config file.
    Returns:
        dict: Loaded configuration dictionary.
    Raises:
        ValueError: If any required field is missing.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Check for required fields
    required_fields = ['train_paths', 'val_paths', 'test_paths', 'batch_size', 'epochs', 'learning_rate']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    return config

def main():
    """
    Main entry point for training and evaluation.
    Parses command-line arguments, loads config, sets device, and starts sweep.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate PCS classifier.")
    parser.add_argument('--config', required=True, help='Path to JSON config file')
    args = parser.parse_args()
    config = load_config(args.config)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sweep(config)

if __name__ == "__main__":
    main()



