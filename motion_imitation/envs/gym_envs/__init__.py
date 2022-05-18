"""Setup such that environment can be created using gym.make()."""
import gym
from gym.envs.registration import make, registry, spec
from motion_imitation.envs.gym_envs.a1_gym_env import A1GymEnv


def register(env_id, *args, **kvargs):
    if env_id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(env_id, *args, **kvargs)


register(
    env_id="A1GymEnv-v0",
    entry_point="motion_imitation.envs.gym_envs:A1GymEnv",
    max_episode_steps=1000,
    reward_threshold=1000.0,
)
