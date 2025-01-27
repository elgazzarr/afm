from dotenv import load_dotenv
load_dotenv()
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import jax.random as jrandom
import flax.nnx as nnx
from models.flow import Flow
from trainers.flow_train import FlowTrainer
from trainers.flow_train_stf import FlowTrainerSTF
from trainers.eval_gluonts import JaxPredictor
from utils.visualization import Visualizer
import orbax.checkpoint as ocp
import jax
import os
import jax.numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer
from data.gluonts_loaders import get_dataset_loader, _to_dataframe
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Flow Matching for Time Series')
    
    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default='electricity_nips',
                      help='Name of the dataset')
    parser.add_argument('--context_ratio', type=float, default=2.0,
                      help='Context ratio for dataset')
    # Training parameters
    parser.add_argument('--seed', type=int, default=1,
                      help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--training_steps', type=int, default=10000,
                      help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--val_interval', type=int, default=500,
                      help='Validation interval')
    parser.add_argument('--patience', type=int, default=15,
                      help='Patience for early stopping')
    parser.add_argument('--student_forcing', type=bool, default=False,)
    
    # Model architecture
    parser.add_argument('--h_dim', type=int, default=64,
                      help='Hidden dimension')
    parser.add_argument('--t_dim', type=int, default=32,
                      help='Time dimension')
    parser.add_argument('--n_rnn_layers', type=int, default=3,
                      help='Number of RNN layers')
    parser.add_argument('--rnn_hidden', type=int, default=64,
                      help='RNN hidden dimension')
    parser.add_argument('--n_residual_blocks', type=int, default=8,
                      help='Number of residual blocks')
    parser.add_argument('--residual_channels', type=int, default=8,
                      help='Number of residual channels')
    parser.add_argument('--residual_hidden', type=int, default=64,
                      help='Residual hidden dimension')
    
    # Inference parameters
    parser.add_argument('--mcmc_samples', type=int, default=100,
                      help='Number of MCMC samples')
    parser.add_argument('--nfe', type=int, default=100,
                      help='Number of function evaluations')
    
    # Logging parameters
    parser.add_argument('--wandb_project', type=str, default='flowtime',
                      help='W&B project name')
    parser.add_argument('--log_interval', type=int, default=500,
                      help='Logging interval')
    parser.add_argument('--save_model', action='store_true',
                      help='Whether to save model checkpoints')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, config=vars(args), group=args.dataset_name)
    config = wandb.config
    
    #print("jax devices: ", jax.devices())
    
    # Set random seed
    rngs = nnx.Rngs(config.seed)
    rngs_model = nnx.Rngs(rngs())  # Get new key from base rngs for model
    rngs_train = nnx.Rngs(rngs())  # Get another new key for training
    rngs_test = nnx.Rngs(rngs())   # Get another new key for testing
    
    # Initialize dataset and dataloader
    print("Loading dataset")
    train_loader, val_loader, test_data, dims, scaler = get_dataset_loader(
        config.dataset_name, 
        config.batch_size, 
        config.context_ratio
    )
    
    # Initialize model
    print("Initializing model...")
    model = Flow(
        dim_x=dims["data_dim"],
        dim_c=dims["cov_dim"],
        dim_h=config.h_dim,
        dim_t=config.t_dim,
        n_rnn_layers=config.n_rnn_layers,
        rnn_hidden=config.rnn_hidden,
        residual_layers=config.n_residual_blocks,
        residual_channels=config.residual_channels,
        residual_hidden=config.residual_hidden,
        rngs=rngs_model,
        scaler=scaler
    )
    
    # Initialize trainer
    if config.student_forcing:
        print("Initializing student forcing trainer")
        trainer = FlowTrainerSTF(config, model, train_loader, val_loader, rngs_train)
    else:
        print("Initializing trainer")
        trainer = FlowTrainer(config, model, train_loader, val_loader, rngs_train)
    
    # Train model
    print("Training model")
    trained_model = trainer.train()
    
    # Initialize evaluator
    print("Evaluating model......")
    predictor = JaxPredictor(
        config,
        trained_model,
        test_data.prediction_length,
        int(test_data.prediction_length * config.context_ratio),
        config.batch_size,
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
    
    # Log metrics
    metrics = {
        "CRPS": agg_metric["mean_wQuantileLoss"],
        "ND": agg_metric["ND"],
        "NRMSE": agg_metric["NRMSE"],
        "CRPS-Sum": agg_metric["m_sum_mean_wQuantileLoss"],
        "ND-Sum": agg_metric["m_sum_ND"],
        "NRMSE-Sum": agg_metric["m_sum_NRMSE"]
    }
    
    # Print and log metrics
    for name, value in metrics.items():
        print(f"{name}: {value}")
    wandb.log(metrics)

if __name__ == "__main__":
    main()
