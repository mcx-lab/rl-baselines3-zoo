"""Wrapper to make the a1 environment suitable for OpenAI gym."""
import gym
from blind_walking.envs import env_builder
from blind_walking.envs.env_wrappers.observation_dictionary_to_array_wrapper import ObservationDictionaryToArrayWrapper
from blind_walking.robots import a1, robot_config


class A1GymEnv(gym.Env):
    """A1 environment that supports the gym interface."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, action_limit=(0.5, 0.5, 0.5), render=False, on_rack=False, **kwargs):
        self._env = env_builder.build_regular_env(
            a1.A1,
            motor_control_mode=robot_config.MotorControlMode.POSITION,
            enable_rendering=render,
            action_limit=action_limit,
            on_rack=on_rack,
            **kwargs,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()

    def render(self, mode):
        return self._env.render(mode)

    def __getattr__(self, attr):
        return getattr(self._env, attr)


class A1BlindWalkingBulletEnv(gym.Env):
    """A1 environment that supports the gym interface."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, action_limit=(0.5, 0.5, 0.5), render=False, on_rack=False, **kwargs):
        from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
        from blind_walking.envs.sensors import environment_sensors

        self._env = env_builder.build_regular_env(
            a1.A1,
            motor_control_mode=robot_config.MotorControlMode.POSITION,
            enable_rendering=render,
            action_limit=action_limit,
            on_rack=on_rack,
            env_sensor_list=[
                environment_sensors.LastActionSensor(num_actions=a1.NUM_MOTORS),
            ],
            obs_wrapper=obs_array_wrapper.ObservationDictionaryToArrayWrapper,
            **kwargs,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()

    def render(self, mode):
        return self._env.render(mode)

    def __getattr__(self, attr):
        return getattr(self._env, attr)
