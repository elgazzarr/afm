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



@hydra.main(version_base="1.2", config_path="configs", config_name="base_config.yaml")
def main(config: DictConfig):
    """Main training function."""

    wandb_config = OmegaConf.to_container(config, resolve=True)

    print("jax devices: ", jax.devices())
    #setup wandb
    wandb.init(project="time_flow", config=wandb_config, name=config.experiment.name)
    
    # Set random seed
    key = jrandom.PRNGKey(config.training.seed)
    rngs = nnx.Rngs(config.training.seed)
    rngs_model = nnx.Rngs(rngs())  # Get new key from base rngs for model
    rngs_train = nnx.Rngs(rngs())  # Get another new key for training    
    
    # Initialize dataset and dataloader
    dataset = LotkaVolterraDataset(config, key)
    dataloader = TimeSeriesDataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    
    # Initialize model
    model = Flow(
        dim_x=config.model.data_dim,
        dim_c=config.model.covariates_dim,
        dim_h=config.model.hidden_dim,
        dim_t=config.model.time_dim,
        dim_hidden=config.model.hidden_dim,
        hidden_blocks=config.model.hidden_blocks,
        rngs=rngs_model
    )
    
    # Initialize trainer
    trainer = FlowTrainer(config, model, dataloader, rngs_train)
    
    # Train model
    trainer.train()
    
    # Evaluate and visualize result`s
    evaluator = FlowEvaluator(model, config.inference.nfe)
    visualizer = Visualizer()
    
    # Generate multiple samples and their predictions
    num_examples = 10
    sample_keys = jrandom.split(key, num_examples)  # Split key for each example
    batch_sample = jax.vmap(dataloader.sample)
    histories, covariates, true_futures = batch_sample(sample_keys)

    # Generate predictions for each sample
    eval_keys = jrandom.split(key, num_examples * config.inference.mcmc_samples)
    eval_keys = eval_keys.reshape((num_examples, config.inference.mcmc_samples, -1))

    batch_predict = jax.vmap(
        lambda k, h, c, tf: jax.vmap(evaluator.predict_trajectory, in_axes=(0, None, None, None))(k, h, c, tf.shape[0])
    )
    predictions = batch_predict(eval_keys, histories, covariates, true_futures)

    # Plot and log each result
    for i in range(num_examples):
        fig = visualizer.plot_trajectories(
            dataset.ts, 
            histories[i], 
            predictions[i], 
            true_futures[i], 
            channel_names=['prey', 'predator'],
            confidence_interval=0.90
        )
        wandb.log({f"Sample {i+1}": wandb.Image(fig)})

    metrics = evaluator.compute_metrics(true_futures, jnp.mean(predictions, axis=1))
    wandb.log(metrics)
    
    if config.logging.save_model:
        
        checkpointer = PyTreeCheckpointer()
        params, _ = nnx.split(model)
        
        save_dir = os.path.join(wandb.run.dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving model to {save_dir}")
        checkpointer.save(save_dir, params, force=True)

if __name__ == "__main__":
    main()