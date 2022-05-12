# Guide to Configure A1GymEnv

## Configuring Observations
Observations are read from sensors, which are specified in `blind_walking/envs/env_builder.py`.  
Sensors implementation can be found in `blind_walking/envs/sensors/`.  
Custom sensors can be written in `environment_sensors.py` or `robot_sensors.py`.

## Configuring Reward Function
Reward function is based on the task specified in `blind_walking/envs/env_builder.py`.  
Reward functions implementation can be found in `blind_walking/envs/env_wrappers/`.

## Generating Target Positions
Run this to generate `target_positions.csv` and `target_positions.png`:
```
python blind_walking/envs/env_wrappers/generate_target_positions.py
```
`target_positions.csv` is a csv file that contains the absolute x, y target coordinates for the robot to follow.  
`target_positions.png` is a plot of the target positions generated for better visualisation.

## Task for robot to follow target positions
1. Add `environment_sensors.TargetPositionSensor()` to the list of env_sensors when creating the environment in `env_builder.py`.
2. Set `task = forward_task_pos.ForwardTask()` in `env_builder.py`.
3. Generate target positions from the instructions above.
4. Now you are ready to start training the robot!