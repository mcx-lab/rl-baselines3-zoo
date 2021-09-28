import argparse
import csv
import glob
import importlib
import os
import sys
import numpy as np
import torch as th
import yaml
import gym
from zipfile import ZipFile
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import obs_as_tensor

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from blind_walking.net.adapter import Adapter


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="A1GymEnv-v0")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--no-save", action="store_true", default=False, help="Do not save the adapter and stats (useful for tests)")
    parser.add_argument("-i", "--trained-agent-folder", help="Path to a pretrained agent folder to continue training", default="", type=str)
    parser.add_argument("--load-best", action="store_true", default=False, help="Load best model instead of last model if available")
    parser.add_argument("--load-checkpoint",
                        type=int,
                        help="Load checkpoint instead of last model if available, "
                        "you must pass the number of timesteps corresponding to it")
    parser.add_argument("--load-last-checkpoint",
                        action="store_true",
                        default=False,
                        help="Load last checkpoint instead of last model if available")
    parser.add_argument("--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument("--gym-packages",
                        type=str,
                        nargs="+",
                        default=[],
                        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor")
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if args.load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_id}, path: {log_path}")

        def step_count(checkpoint_path: str) -> int:
            # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
            return int(checkpoint_path.split("_")[-2])

        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        found = True

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

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

    # Load trained model
    trained_model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    # Get actor-critic policy which contains the feature extractor and ppo
    trained_policy = trained_model.policy
    trained_policy.eval()
    trained_feature_encoder = trained_policy.features_extractor
    trained_base_policy = trained_policy.mlp_extractor
    trained_base_policy_action = trained_policy.action_net
    trained_base_policy_value = trained_policy.value_net

    # Reset environment
    obs = env.reset()

    # Hyperparameters
    # TODO - change to hyperparameters config
    n_epochs = 1000
    n_timesteps = int(2e3)
    n_minibatch = 4
    minibatch_size = int(n_timesteps/n_minibatch)
    learning_rate = 5e-4

    # Create adapter model
    adapter = Adapter(trained_policy.observation_space,
                      output_size=trained_feature_encoder.mlp_output_size)
    criterion = th.nn.MSELoss()
    optimizer = th.optim.Adam(adapter.parameters(), lr=learning_rate)

    # Load from pretrained adapter if given
    if args.trained_agent_folder:
        adapter_path = os.path.join(args.trained_agent_folder, 'adapter.pth')
        adapter.load_state_dict(th.load(adapter_path))
        adapter.train()
        optimizer_path = os.path.join(args.trained_agent_folder, 'adapter.optimizer.pth')
        optimizer.load_state_dict(th.load(optimizer_path))

    # File creation for saving
    if not args.no_save:
        adapter_folder = os.path.join(log_path, f'{env_id}_adapter')
        if not os.path.exists(adapter_folder):
            os.mkdir(adapter_folder)
        statsfile = os.path.join(adapter_folder, 'stats.csv')
        with open(statsfile, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['epoch', 'running loss'])

    # Train adapter model
    try:
        for epoch in range(n_epochs):
            obs = env.reset()
            running_loss = 0
            for i in range(n_timesteps):
                obs = obs_as_tensor(obs, trained_model.device)

                # Forward
                predicted_extrinsics = adapter(obs)
                target_extrinsics = trained_feature_encoder(obs)

                policy_output = trained_base_policy(predicted_extrinsics)
                action = trained_base_policy_action(policy_output[0])
                value = trained_base_policy_value(policy_output[1])

                # Clip and perform action
                clipped_action = action.detach().cpu().numpy()
                if isinstance(trained_model.action_space, gym.spaces.Box):
                    clipped_action = np.clip(clipped_action,
                                             trained_model.action_space.low,
                                             trained_model.action_space.high)
                obs, reward, done, infos = env.step(clipped_action)

                # Accumulate loss
                loss = criterion(predicted_extrinsics, target_extrinsics)
                loss.backward()
                running_loss += loss

                # Optimize per mini batch
                if (i+1) % minibatch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            # Print stats
            if not args.no_save:
                with open(statsfile, 'a') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([epoch, running_loss.item()])
            if (epoch+1) % 10 == 0:
                print(f'epoch {epoch}, running loss: {running_loss}')
    except KeyboardInterrupt:
        # Allow saving of model when training is interrupted
        pass

    # Release resources
    env.close()

    # Save adapter model
    if not args.no_save:
        print('saving model...')
        adapter_path = os.path.join(adapter_folder, 'adapter.pth')
        th.save(adapter.state_dict(), adapter_path)
        adapter_optimizer_path = os.path.join(adapter_folder, 'adapter.optimizer.pth')
        th.save(optimizer.state_dict(), adapter_optimizer_path)

if __name__ == "__main__":
    main()
