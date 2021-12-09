#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
import yaml
from stable_baselines3.common.utils import set_random_seed
import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_saved_hyperparams


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--log-path", help="Path to folder containing pre-trained model", type=str, default="./")
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    args = parser.parse_args()

    env_id = "A1GymEnv-v0"
    algo = "ppo"
    log_path = args.log_path

    # ######################### Create environment ######################### #

    set_random_seed(args.seed)
    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    obs = env.reset()

    # ######################### Load model ######################### #

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    model_path = os.path.join(log_path, f"{env_id}.zip")
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects)

    # ######################### Enjoy Loop ######################### #

    state = None
    for _ in range(args.n_timesteps):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, infos = env.step(action)

    env.close()


if __name__ == "__main__":
    main()