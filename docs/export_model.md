# Deploy Model to Gazebo

## Quickstart

Train a model: 
```
python train.py --algo ppo --env A1GymEnv-v0 -f logs
```

Deploy model and log statistics:

```
python scripts/enjoy_with_logging.py --algo ppo --env A1GymEnv-v0 -f logs
```

Export model by running `notebooks/export_model.ipynb`. 



