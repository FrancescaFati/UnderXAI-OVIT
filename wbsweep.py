import wandb 
import main

def sweep_agent(config=None):
    with wandb.init(
        config=config,
        ):
        config = wandb.config
        print(config)

        main.sweep(config)


sweep_agent()