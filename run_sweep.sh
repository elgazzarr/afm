#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gm

# Initialize sweep (only run this once)
/home/artiintel/ahmelg/.conda/envs/gm/bin/python -m wandb sweep sweep_config.yaml

# Run sweep agent (run this multiple times for parallel runs)
/home/artiintel/ahmelg/.conda/envs/gm/bin/python -m wandb agent YOUR_USERNAME/flowtime/SWEEP_ID