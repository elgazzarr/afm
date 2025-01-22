from dotenv import load_dotenv
load_dotenv()
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import jax.random as jrandom
import flax.nnx as nnx
from models.flow import Flow
from trainers.flow_train import FlowTrainer
from trainers.eval_gluonts import JaxPredictor
from utils.visualization import Visualizer
import orbax.checkpoint as ocp
import jax
import os
import jax.numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer
from data.gluonts_loaders import get_dataset_loader, _to_dataframe

@hydra.main(version_base="1.2", config_path="configs", config_name="gluonts_config.yaml")
def main(config: DictConfig):
    """Main training function with WandB sweep support."""
    # Initialize wandb with sweep configuration
    wandb_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(project=config.logging.wandb.project, config=wandb_config)
    
    # Update config with wandb sweep parameters
    sweep_config = wandb.config
    for key, value in sweep_config.items():
        if '.' in key:
            # Handle nested config updates
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], value)
    
    print("jax devices: ", jax.devices())
    
    # Set random seed
    rngs = nnx.Rngs(config.training.seed)
    rngs_model = nnx.Rngs(rngs())
    rngs_train = nnx.Rngs(rngs())
    rngs_test = nnx.Rngs(rngs())
    
    # Initialize dataset and dataloader
    print("Loading dataset")
    dataset = config.experiment.dataset_name
    train_loader, val_loader, test_data, dims, scaler = get_dataset_loader(
        dataset, 
        config.training.batch_size, 
        config.experiment.context_ratio
    )
    
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
    trained_model = trainer.train()
    
    # Initialize evaluator
    print("Evaluating model ......")
    predictor = JaxPredictor(
        config, 
        trained_model, 
        test_data.prediction_length,
        test_data.prediction_length * config.experiment.context_ratio,
        config.training.batch_size,
        scaler=scaler,
        rngs=rngs_test
    )
    
    forecasts_iter = predictor.predict(test_data)
    targets_iter = map(_to_dataframe, test_data)
    agg_metric, item_metrics = predictor.evaluate(
        targets_iter, 
        forecasts_iter, 
        num_series=len(test_data)
    )
    
    # Log metrics to wandb
    wandb.log({
        "CRPS": agg_metric["mean_wQuantileLoss"],
        "ND": agg_metric["ND"],
        "NRMSE": agg_metric["NRMSE"],
        "CRPS-Sum": agg_metric["m_sum_mean_wQuantileLoss"],
        "ND-Sum": agg_metric["m_sum_ND"],
        "NRMSE-Sum": agg_metric["m_sum_NRMSE"]
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()