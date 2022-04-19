"""Wrapper for A1 skating env."""
import gym
from blind_walking.envs import env_builder
from blind_walking.envs.env_wrappers.observation_dictionary_to_array_wrapper import ObservationDictionaryToArrayWrapper
from blind_walking.envs.gym_envs import a1_gym_env
from blind_walking.envs.tasks import rollerskating_task
from blind_walking.robots import a1_wheeled, robot_config
from numpy import roll


class A1SkatingEnv(a1_gym_env.A1GymEnv):
    def __init__(self, action_limit=(0.5, 0.5, 0.5), render=False, on_rack=False, **kwargs):
        self._env = env_builder.build_regular_env(
            a1_wheeled.A1,
            motor_control_mode=robot_config.MotorControlMode.POSITION,
            enable_rendering=render,
            action_limit=action_limit,
            on_rack=on_rack,
            task=rollerskating_task.RollerskatingTask() ** kwargs,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
