"""GluonTS benchmark pipeline (Table 4 of the paper). Faithful port of git revision f4131fe^ (main.py). Deviations from the original, kept minimal: (1) imports fixed to relative imports and unused imports (dotenv/hydra/omegaconf/orbax/Visualizer/jax/jnp/jrandom/os usages) dropped; (2) wandb is optional — logging is disabled unless ``--use-wandb`` is passed (the original always called ``wandb.init``); (3) the config namespace is the parsed args directly (the original read ``wandb.config``, which mirrors ``vars(args)``). The dataset/model/evaluation logic is unchanged. See REPRODUCIBILITY.md.

Usage (after ``pip install -e ".[gluonts]"``):
    python -m afm.gluonts_pipeline.run --dataset_name exchange_rate_nips
"""
import wandb
import flax.nnx as nnx
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from .flow import Flow
from .trainer import FlowTrainer
from .predictor import JaxPredictor
from .loaders import get_dataset_loader, _to_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description='Flow Matching for Time Series')

    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default='exchange_rate_nips',
                        help='Name of the dataset', choices=['traffic_nips', 'electricity_nips', 'solar_nips', 'exchange_rate_nips', 'wiki-rolling_nips'])
    parser.add_argument('--context_ratio', type=float, default=2.0,
                        help='Context ratio for dataset')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--training_steps', type=int, default=1000,
                        help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')

    # Model architecture (Flow)
    parser.add_argument('--h_dim', type=int, default=128,
                        help='Hidden dimension of the RNN')
    parser.add_argument('--t_dim', type=int, default=64,
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
    parser.add_argument('--data_scaling', type=bool, default=False,
                        help='Whether to mean scale data')
    parser.add_argument('--context_scaling', type=bool, default=True,
                        help='Whether to mean scale context')

    # Inference parameters
    parser.add_argument('--mcmc_samples', type=int, default=100,
                        help='Number of MCMC samples')
    parser.add_argument('--nfe', type=int, default=100,
                        help='Number of function evaluations')

    # Logging parameters
    parser.add_argument('--wandb_project', type=str, default='flowtime',
                        help='W&B project name')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='Logging interval')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model checkpoints')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable wandb logging (default: disabled)')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Initialize wandb (disabled by default so the pipeline runs offline;
    # pass --use-wandb to log to the wandb project as in the paper's runs)
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        group=args.dataset_name,
        mode=(os.environ.get("WANDB_MODE") if args.use_wandb else "disabled"),
    )
    config = wandb.config if args.use_wandb else args

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
        config.context_ratio,
        scale=config.data_scaling
    )

    flow_model = Flow(
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
        scaler=scaler,
        context_scaling=config.context_scaling,
    )

    # Train model
    print("Training Flow ...")
    trainer = FlowTrainer(config, flow_model, train_loader, val_loader, rngs_train)
    trained_flow = trainer.train()
    print("*** Flow training complete ***")

    # Initialize evaluator
    print("Evaluating model......")
    predictor = JaxPredictor(
        config,
        trained_flow,
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

    if wandb.run is not None and args.use_wandb:
        wandb.log(metrics)
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
