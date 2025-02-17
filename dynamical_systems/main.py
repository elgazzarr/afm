import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import flax.nnx as nnx
from config import ModelConfig
from systems import create_system
from models import ARFlow, NoARFlow
from training import prepare_data, ARTrainer, NoARTrainer
from visualization import plot_loss, plot_trajectories
import diffrax as dfx 
from seq_priors import generate_brownian_timeseries
import argparse
from sample import sample_flow

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating Autoregressive vs Non-Autoregressive Flow matching for forecasting dynamical systems')

    parser.add_argument('--system', type=str, default='lorenz',
                        options=['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator', 'brusselator'],
                      help='Name of the dynamical system')
    parser.add_argument('--stochastic', type=bool, default=False,
                        help='Whether the system is stochastic')
    parser.add_argument('--t1', type=float, default=20.0)
    parser.add_argument('--num_timepoints', type=int, default=200,
                        help='Number of timepoints')
    parser.add_argument('--history_size', type=int, default=50,
                        help='History size')
    parser.add_argument('--prediction_size', type=int, default=100,
                        help='Prediction size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    # Set random seed
    key = jrandom.PRNGKey(0)

    # Set up configuration
    config = ModelConfig(
        system=args.system,
        num_timepoints=args.num_timepoints,
        history_size=args.history_size,
        prediction_size=args.prediction_size,
        t0=0.0,
        t1=args.t1,
        n_train_test_samples=1024,
        batch_size=128
    )

    # Create system
    system = create_system(config.system, config.stochastic)
    (train_loader, train_trajectories), (test_loader, test_trajectories) = prepare_data(system, config, key)

    key, key_ar, key_noar = jrandom.split(key, 3)

    # AR Model
    ar_model = ARFlow(
        dim_x=system.data_dim,
        dim_c=config.covariates_dim,
        dim_h=config.hidden_dim,
        dim_t=config.time_dim,
        dim_hidden=config.hidden_dim,
        hidden_blocks=config.hidden_blocks,
        n_rnn_layers=config.n_rnn_layers,
        rnn_hidden=config.rnn_hidden,
        rngs=nnx.Rngs(key_ar)
    )

    # NoAR Model
    noar_model = NoARFlow(
        dim_x=system.data_dim,
        dim_c=config.covariates_dim,
        dim_h=config.hidden_dim,
        dim_t=config.time_dim,
        dim_hidden=config.hidden_dim,
        hidden_blocks=config.hidden_blocks,
        n_rnn_layers=config.n_rnn_layers,
        rngs=nnx.Rngs(key_noar)
    )

        # Train NoAR model
    ar_trainer = ARTrainer(ar_model, learning_rate=config.learning_rate)
    noar_trainer = NoARTrainer(noar_model, learning_rate=config.learning_rate)

    # Train models
    ar_losses = ar_trainer.train(train_loader, test_loader, num_epochs=args.num_epochs)
    noar_losses = noar_trainer.train(train_loader, test_loader, num_epochs=args.num_epochs)

    # Generate predictions
    def sample_trajectories(model, key, num_samples=50, nfe=50):
        history, covariates, true_future, true_exp = next(iter(test_loader))
        true_fe = jnp.concatenate([true_future, true_exp], axis=1)
        predictions = sample_flow(model, key, history, covariates, n_steps_flow=nfe, n_steps_pred=config.prediction_size+config.extrapolation_size, n_samples=num_samples)
        return jnp.stack(predictions), history, true_fe
    
    def evaluate_predictions(model, key, test_loader, nfe=50):
        crps = []
        for i, (history, covariates, true_future, true_exp) in enumerate(test_loader):
            predictions = sample_flow(model, key, history, covariates, n_steps_flow=nfe, n_steps_pred=config.prediction_size+config.extrapolation_size, n_samples=50)
            crps = 
        
    
    ar_predictions, ar_history, ar_true = sample_trajectories(ar_model, key)
    noar_predictions, noar_history, noar_true = sample_trajectories(noar_model, key)
    noar_predictions = jnp.squeeze(noar_predictions, axis=2)

    for i in range(5):
        plot_trajectories(config.ts, ar_history[i], ar_predictions[:,i], ar_true[i], config.extrapolation_size, title='AR Model Predictions', 
                          save_path=f'figures/{config.system}/ar_predictions_{i}.pdf')
        plot_trajectories(config.ts, noar_history[i], noar_predictions[:,i], noar_true[i], config.extrapolation_size, title='NoAR Model Predictions',
                           save_path=f'figures/{config.system}/noar_predictions_{i}.pdf')

    return


if __name__ == '__main__':
    main()


