import random
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from blind_walking.envs.utilities import env_randomizer_base

@dataclass 
class RMAEnvRandomizerConfig:
    resample_probability: float
    controller_Kp_lower_bound: float
    controller_Kp_upper_bound: float 
    controller_Kd_lower_bound: float
    controller_Kd_upper_bound: float
    motor_strength_ratios_lower_bound: float
    motor_strength_ratios_upper_bound: float

config_registry = {
    'no_var_train': RMAEnvRandomizerConfig(
        resample_probability=0,
        controller_Kp_lower_bound=55, 
        controller_Kp_upper_bound=55, 
        controller_Kd_lower_bound=0.6,
        controller_Kd_upper_bound=0.6,
        motor_strength_ratios_lower_bound=1.0,
        motor_strength_ratios_upper_bound=1.0
    ),
    'rma_easy_train': RMAEnvRandomizerConfig(
        resample_probability=0.004,
        controller_Kp_lower_bound=50, 
        controller_Kp_upper_bound=60, 
        controller_Kd_lower_bound=0.4,
        controller_Kd_upper_bound=0.8,
        motor_strength_ratios_lower_bound=0.9,
        motor_strength_ratios_upper_bound=1.0
    ),
    'rma_hard_train': RMAEnvRandomizerConfig(
        resample_probability=0.01,
        controller_Kp_lower_bound=45, 
        controller_Kp_upper_bound=65, 
        controller_Kd_lower_bound=0.3,
        controller_Kd_upper_bound=0.9,
        motor_strength_ratios_lower_bound=0.88,
        motor_strength_ratios_upper_bound=1.0
    )   
}

test_configs = {}
for train_config_name, config in config_registry.items():
    test_config_name = "_".join(train_config_name.split("_")[:-1] + ['test'])
    test_config = deepcopy(config_registry[train_config_name])
    # Set the resample probability to 1 so that various envs will be seen in evaluation
    test_config.resample_probability = 1
    test_configs[test_config_name] = test_config

config_registry.update(test_configs)

class RMAEnvRandomizer(env_randomizer_base.EnvRandomizerBase):
    """ A randomizer that perturbs the A1 gym env according to RMA paper """
    
    def __init__(self, config):
        self.config = config

    def randomize_env(self, env):
        robot = env._robot
        if np.random.uniform() < self.config.resample_probability:
            Kp = np.random.uniform(self.config.controller_Kp_lower_bound, self.config.controller_Kp_upper_bound)
            Kd = np.random.uniform(self.config.controller_Kd_lower_bound, self.config.controller_Kd_upper_bound)
            motor_strength_ratio = np.random.uniform(self.config.motor_strength_ratios_lower_bound, 
                self.config.motor_strength_ratios_upper_bound)
            robot.SetMotorGains(Kp, Kd)
            robot.SetMotorStrengthRatio(motor_strength_ratio)