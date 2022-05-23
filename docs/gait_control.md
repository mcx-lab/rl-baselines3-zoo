# Gait Control Code Documentation

## Quickstart

Run with default settings: 
```
python train.py --algo ppo --env A1GymEnv-v0 -f logs
```

By default, the configured gait is a walking gait with frequency 1.5Hz and duty factor of 0.5. The policy will observe reference foot contacts for only the current timestep. 

## Training with Gait Randomization

In order for the policy to respond well to different gaits, it is necessary to make the policy imitate a variety of gaits during training. Gait randomization can be set up by configuring `--env-kwargs` in the train script. 

When training with multiple gaits, it is important for the policy to anticipate changes in gait ahead of time, i.e. predictive control. This can be done by configuring the `obs_steps_ahead` in `--env-kwargs`. 

Example of gait randomization with predictive control: 

```
python train.py --algo ppo --env A1GymEnv-v0 -f logs \
--env-kwargs \
  gait_names:["'walk'","'trot'"]
  gait_frequency_upper:1.9
  gait_frequency_lower:1.1
  duty_factor_upper:0.85
  duty_factor_lower:0.55
  obs_steps_ahead:(0,1,2,10,50)
```

This will configure randomization of gait on each episode as follows:
- Gait will be uniformly randomly chosen to be a walk or a trot
- Gait frequency will be uniformly randomly chosen in (1.1, 1.9)
- Duty factor will be uniformly randomly chosen in (0.55, 0.85)
- Policy will observe (0, 1, 2, 10, 50) steps ahead. With env time step at 0.01s, policy sees up to 0.5 seconds ahead. Choice of timesteps has not been explored yet. 

## Evaluating with set gaits


Evaluation on a fixed gait can be done as follows:

```
python scripts/enjoy_with_logging.py --algo ppo --env A1GymEnv-v0 -f logs --no-render \
--env-kwargs \
  gait_names:["'walk'","'trot'"]
  gait_frequency_upper:1.5
  gait_frequency_lower:1.5
  duty_factor_upper:0.75
  duty_factor_lower:0.75
```

The enjoy script re-uses the same keyword arguments used when training and then overwrites with the ones provided to the script itself. Hence we overwrite the desired gait patterns to gaits we wish to test on. However, there is no need to set `obs_steps_ahead` as that remains the same. 

