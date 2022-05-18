"""Wrapper to make the a1 environment suitable for OpenAI gym."""
import gym
from motion_imitation import motion_data_dir
from motion_imitation.envs import env_builder
from motion_imitation.robots import a1, robot_config


class A1GymEnv(gym.Env):
    """A1 environment that supports the gym interface."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        imit_mode="full",
        motion_files=[str(motion_data_dir / "dog_pace.txt")],
        tar_frame_steps=[1, 2, 10, 30],
        enable_randomizer=False,
    ):
        self._env = env_builder.build_imitation_env(
            imit_mode=imit_mode,
            motion_files=motion_files,
            tar_frame_steps=tar_frame_steps,
            enable_randomizer=enable_randomizer,
            enable_rendering=False,
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
