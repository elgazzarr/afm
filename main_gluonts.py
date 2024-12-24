from dotenv import load_dotenv
load_dotenv()
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import jax.random as jrandom
import flax.nnx as nnx
from models.flow import Flow
from data.datasets import LotkaVolterraDataset
from data.dataloader import TimeSeriesDataLoader
from trainers.flow_train import FlowTrainer
from trainers.flow_eval import FlowEvaluator
from utils.visualization import Visualizer
import orbax.checkpoint as ocp
import jax
import os
import jax.numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer
from data.gluonts_loaders import get_dataset_loader



@hydra.main(version_base="1.2", config_path="configs", config_name="gluonts_config.yaml")
def main(config: DictConfig):
    """Main training function."""

    wandb_config = OmegaConf.to_container(config, resolve=True)

    print("jax devices: ", jax.devices())
    #setup wandb
    wandb.init(project="time_flow", config=wandb_config, name=config.experiment.dataset_name)
    
    # Set random seed
    key = jrandom.PRNGKey(config.training.seed)
    rngs = nnx.Rngs(config.training.seed)
    rngs_model = nnx.Rngs(rngs())  # Get new key from base rngs for model
    rngs_train = nnx.Rngs(rngs())  # Get another new key for training    
    
    # Initialize dataset and dataloader
    print("Loading dataset")
    dataset = config.experiment.dataset_name
    train_loader, val_loader, test_loader, dims, scaler = get_dataset_loader(dataset, config.training.batch_size, config.experiment.context_ratio)

    # Initialize model
    print("Initializing model...")
    model = Flow(
        dim_x=dims["data_dim"],
        dim_c=dims["cov_dim"],
        dim_h=config.model.h_dim,
        dim_t=config.model.t_dim,
        n_rnn_layers=config.model.n_rnn_layers,
        rnn_hidden=config.model.rnn_hidden,
        residual_layers=config.model.n_residual_blocks,
        residual_channels=config.model.residual_channels,
        residual_hidden=config.model.residual_hidden,
        rngs=rngs_model,
        scaler=scaler
    )
    
    # Initialize trainer
    print("Initializing trainer")
    trainer = FlowTrainer(config, model, train_loader, val_loader, rngs_train)
    
    # Train model
    print("Training model")
    trainer.train()

if __name__ == "__main__":
    main()