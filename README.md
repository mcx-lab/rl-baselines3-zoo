# Gait Imitation: A Framework for Imitaton-Guided RL for Quadruped Robots

## Install

To install only the core dependencies:
```
python -m pip install -e .
```
To install developer dependencies:
```
python -m pip install -e .[dev]
pre-commit install
```

## WandB Setup

WandB logging can be enabled. 
To use WandB logging, create a WandB account. Then run:
```
python -m pip install wandb
wandb login
```
When prompted, enter the API token from your account. 
This enables WandB logging in `train.py` with the flag `--use-wandb`  

By default it looks for an entity `mcx-lab` and logs to a project `terrain-aware-locomotion`. 
See `utils.ExperimentManager.setup_experiment()` for details on how the logger is configured. 