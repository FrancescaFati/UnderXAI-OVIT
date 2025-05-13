import wandb
import torch
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from pcs_dataset import PCSDataset
from engine import Engine
from model import Vit_Classifier, Slice_Contrastive, Volume_Contrastive
from eval import ModelEvaluator
from utilis import detect_aval_cpus
from rich import print
import numpy as np
from torch.utils.data import DataLoader, Subset
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def sweep(config=None):

    with wandb.init(
        #project=config['project_name'],
        #entity="underxai",
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):        
     

        model = Vit_Classifier(config).to(device)
        print("Model loaded")    
        
        train_dataset = PCSDataset(
            data_file=config['train_paths'],
            excluded_years=[2015],
            transform=None,
            is_training=True, 
            seg = False, 
            seg_as_im = False,
            clinical_fe=False,
        )
        val_dataset = PCSDataset(
            data_file=config['val_paths'],
            excluded_years=[2015],
            transform=None,
            is_training=False, 
            seg = False, 
            seg_as_im = False,
            clinical_fe=False,
        )      
        test_dataset = PCSDataset(
            data_file=config['test_paths'],
            excluded_years=[2015],
            transform=None,
            is_training=False, 
            seg = False, 
            seg_as_im = False,
            clinical_fe=False,
        )           
                    
                        
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
        

        engine = Engine(
            model=model, 
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        engine.train(num_epochs=config['epochs'])
    
        # Create models directory
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
        
        print()
        print("[orange3]TEST[/orange3]")
        
        test_metrics = {"test_accuracy": 0.0, "test_f1_score": 0.0}
        
        evaluator = ModelEvaluator(model, device, config=config)
        n_iterations = 1
        n_size = len(test_dataset)
        boostrap_metrics = []
        
        for i in range(n_iterations): 
            
            indices =  np.random.choice(np.arange(n_size), size=n_size, replace=True)
            bootstrap_subset = Subset(test_dataset, indices)
            bootstrap_loader = DataLoader(bootstrap_subset, batch_size=config["batch_size"], shuffle=False)
            
            comprehensive_test_metrics, comprehensive_test_metrics_opthr = evaluator.predict(bootstrap_loader)
            boostrap_metrics.append(comprehensive_test_metrics)
            print(f"{i+1} / {n_iterations}")
            
        bootstrap_df = pd.DataFrame(boostrap_metrics)
        bootstrap_df.to_csv(f"bootstrap_metrics_{wandb.run.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        



