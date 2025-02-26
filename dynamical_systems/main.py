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
import wandb
from metrics import compute_crps, compute_nrmse
import warnings
warnings.filterwarnings("ignore", message="Unable to parse backend config for custom call")



def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating Autoregressive vs Non-Autoregressive Flow matching for forecasting dynamical systems')

    parser.add_argument('--system', type=str, default='lorenz',
                        choices=['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator', 'brusselator', 'fitzhugh_nagumo', 'duffing'],
                      help='Name of the dynamical system')
    parser.add_argument('--stochastic', type=bool, default=True,
                        help='Whether the system is stochastic')
    parser.add_argument('--t1', type=float, default=20.0)
    parser.add_argument('--num_timepoints', type=int, default=200,
                        help='Number of timepoints')
    parser.add_argument('--history_size', type=int, default=75,
                        help='History size')
    parser.add_argument('--prediction_size', type=int, default=75,
                        help='Prediction size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--nfe', type=int, default=20,
                        help='Number of function evaluations')
    parser.add_argument('--wandb_project', type=str, default='flowtime_ds',
                      help='W&B project name')
    return parser.parse_args()

def main():
    args = parse_args()
    key = jrandom.PRNGKey(0)
    wandb.init(project=args.wandb_project, config=vars(args), name=args.system+f"_{args.nfe}")

    # Set up configuration
    config = ModelConfig(
        system=args.system,
        num_timepoints=args.num_timepoints,
        history_size=args.history_size,
        prediction_size=args.prediction_size,
        t0=0.0,
        t1=args.t1,
        n_train_test_samples=1024,
        batch_size=256,
        n_steps_flow=args.nfe,
    )

    # Create system
    system = create_system(config.system, config.stochastic)
    (train_loader, train_trajectories), (test_loader, test_trajectories) = prepare_data(system, config, key)


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
        rngs=nnx.Rngs(key)
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
        rngs=nnx.Rngs(key)
    )

    # Train NoAR model
    ar_trainer = ARTrainer(ar_model, learning_rate=config.learning_rate)
    noar_trainer = NoARTrainer(noar_model, learning_rate=config.learning_rate)

    # Train models
    print('Training AR Model')
    ar_trainer.train(train_loader, test_loader, num_epochs=args.num_epochs)
    print('Training NoAR Model')
    noar_trainer.train(train_loader, test_loader, num_epochs=args.num_epochs*2)
    print('Training complete')
    print('#'*50)

    # Generate predictions
    def sample_trajectories(model, key, num_samples=50, nfe=50):
        history, covariates, true_future, true_exp = next(iter(test_loader))
        true_fe = jnp.concatenate([true_future, true_exp], axis=1)
        predictions = sample_flow(model, key, history, covariates, n_steps_flow=nfe, n_steps_pred=config.prediction_size+config.extrapolation_size, n_samples=num_samples)
        return jnp.stack(predictions), history, true_fe
    
    # Modified evaluation function
    @nnx.jit
    def evaluate_batch(predictions, true_future):
        """Evaluate metrics for a single batch."""
        crps = compute_crps(predictions, true_future)
        nrmse = compute_nrmse(predictions, true_future)
        return crps, nrmse

    def evaluate_predictions(model, key, test_loader, nfe=50):
        crps_pred = []
        nrmse_pred = []
        crps_exp = []
        nrmse_exp = []
        
        for i, (history, covariates, true_future, true_exp) in enumerate(test_loader):
            predictions = sample_flow(model, key, history, covariates, 
                                    n_steps_flow=nfe, 
                                    n_steps_pred=config.prediction_size+config.extrapolation_size, 
                                    n_samples=50)
            #true_future = jnp.concatenate([true_future, true_exp], axis=1)
            if isinstance(model, NoARFlow):
                predictions = jnp.squeeze(predictions, axis=2)
            crps_p, nrmse_p = evaluate_batch(predictions[:, :, :config.prediction_size], true_future)
            crps_e, nrmse_e = evaluate_batch(predictions[:, :, config.prediction_size:], true_exp)
            crps_pred.append(crps_p)
            nrmse_pred.append(nrmse_p)
            crps_exp.append(crps_e)
            nrmse_exp.append(nrmse_e)

        
        return jnp.mean(jnp.stack(crps_pred)), jnp.mean(jnp.stack(nrmse_pred)), jnp.mean(jnp.stack(crps_exp)), jnp.mean(jnp.stack(nrmse_exp))
            
    print('Evaluating models...')
    ar_crps_p, ar_nrmse_p, ar_crps_e, ar_nrmse_e = evaluate_predictions(ar_model, key, test_loader)
    noar_crps, noar_nrmse, noar_crps_e, noar_nrmse_e = evaluate_predictions(noar_model, key, test_loader)
    print(f'AR Model CRPS: {ar_crps_p}, NRMSE: {ar_nrmse_p}, CRPS Exp: {ar_crps_e}, NRMSE Exp: {ar_nrmse_e}')
    print(f'NoAR Model CRPS: {noar_crps}, NRMSE: {noar_nrmse}, CRPS Exp: {noar_crps_e}, NRMSE Exp: {noar_nrmse_e}')

    ar_predictions, ar_history, ar_true = sample_trajectories(ar_model, key)
    noar_predictions, noar_history, noar_true = sample_trajectories(noar_model, key)
    noar_predictions = jnp.squeeze(noar_predictions, axis=2)

    wandb.log({'AR/CRPS_P': ar_crps_p, 'AR/NRMSE_P': ar_nrmse_p, 'AR/CRPS_E': ar_crps_e, 'AR/NRMSE_E': ar_nrmse_e})
    wandb.log({'NoAR/CRPS_P': noar_crps, 'NoAR/NRMSE_P': noar_nrmse, 'NoAR/CRPS_E': noar_crps_e, 'NoAR/NRMSE_E': noar_nrmse_e})

    for i in range(10):
        plot_trajectories(config.ts, ar_history[i], ar_predictions[:,i], ar_true[i], config.extrapolation_size, title='AR Model Predictions', save_path=f'results/{config.system}/ar_predictions_{i}.pdf')
        plot_trajectories(config.ts, noar_history[i], noar_predictions[:,i], noar_true[i], config.extrapolation_size, title='NoAR Model Predictions', save_path=f'results/{config.system}/noar_predictions_{i}.pdf')
    return


if __name__ == '__main__':
    main()


