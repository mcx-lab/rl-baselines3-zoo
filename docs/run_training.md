## Install dependencies

Install dependencies
```
conda create -n rl-baselines3-zoo python=3.8
conda activate rl-baselines3-zoo
# now within the conda environment
python -m pip install -e .[dev]
```

Set up WandB integration
```
wandb login
# Follow the instructions to log in
```

## Run roller skating experiment

```
python train.py \
    --algo ppo \
    --env A1SkatingGymEnv-v0 \
    -f logs \
    --n-timesteps 2000000 \
    --save-freq 100000 \
    --use-wandb \
    --project-name roller_skating \
    --run-name test-flight-0 \
    --tensorboard tensorboard_logs
```