import argparse
import glob
import importlib
import os
import sys
import numpy as np
import torch as th
import yaml
import gym

from stable_baselines3.common.utils import set_random_seed, obs_as_tensor
from stable_baselines3.common.vec_env import VecVideoRecorder

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from blind_walking.net.adapter import Adapter


class Plotter:
    def __init__(self, name: str = "plot"):
        self.data = []
        self.name = name

    def update(self, data: np.ndarray):
        self.data.append(data)

    def plot(self, save_path: str = None):
        if save_path is None:
            save_path = f"{self.name}.png"
        import matplotlib.pyplot as plt

        all_data = np.concatenate(self.data)
        plt.figure()
        t = np.arange(all_data.shape[0])
        for i in range(all_data.shape[1]):
            plt.plot(t, all_data[:, i])
        plt.savefig(save_path)


class HeightmapLogger:
    def __init__(self, name: str = "log"):
        self.data = []
        self.name = name

    def update(self, data: np.ndarray):
        self.data.append(data)

    def save(self, save_path: str = None):
        if save_path is None:
            save_path = self.name
        all_data = np.concatenate(self.data)
        np.save(save_path, all_data)


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ppo",
        type=str,
        required=False,
        choices=list(ALGOS.keys()),
    )
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument(
        "--num-threads",
        help="Number of threads for PyTorch (-1 to use default)",
        default=-1,
        type=int,
    )
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument(
        "--exp-id",
        help="Experiment ID (default: 0: latest, -1: no exp folder)",
        default=0,
        type=int,
    )
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument("--record", action="store_true", default=False, help="Record video")
    parser.add_argument("-o", "--output-folder", help="Video output folder", type=str)
    parser.add_argument(
        "--no-render",
        action="store_true",
        default=False,
        help="Do not render the environment (useful for tests)",
    )
    parser.add_argument("--plot-encoder-output", action="store_true", default=False, help="Log the encoder output to a file")
    parser.add_argument(
        "--adapter",
        action="store_true",
        default=False,
        help="Use adapter module instead of trained model",
    )
    parser.add_argument(
        "--load-best",
        action="store_true",
        default=False,
        help="Load best model instead of last model if available",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic actions",
    )
    parser.add_argument(
        "--norm-reward",
        action="store_true",
        default=False,
        help="Normalize reward if applicable (trained with VecNormalize)",
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    # ######################### Get experiment folder ######################### #

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
        name_prefix = f"final-model-{algo}-{env_id}"
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)
        name_prefix = f"best-model-{algo}-{env_id}"

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)
        name_prefix = f"checkpoint-{args.load_checkpoint}-{algo}-{env_id}"

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
        name_prefix = f"checkpoint-{step_count(model_path)}-{algo}-{env_id}"

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    print(f"Loading {model_path}")

    # ######################### Create environment ######################### #

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
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
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    obs = env.reset()

    # If record video
    if args.record:
        video_folder = args.output_folder
        if video_folder is None:
            if args.adapter:
                video_folder = os.path.join(log_path, "videos_adapter")
            else:
                video_folder = os.path.join(log_path, "videos")
        env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=args.n_timesteps,
            name_prefix=name_prefix,
        )
        env.reset()

    # ######################### Load model ######################### #

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

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    # Get actor-critic policy which contains the feature extractor and ppo
    policy = model.policy
    policy.eval()
    feature_encoder = policy.features_extractor
    base_policy = policy.mlp_extractor
    base_policy_action = policy.action_net
    base_policy_value = policy.value_net
    if args.plot_encoder_output:
        true_extrinsics_plotter = Plotter(name="true_extrinsics")
        heightmap_logger = HeightmapLogger(name="heightmap_sensor_observations")

    if args.adapter:
        # Load adapter module
        adapter_path = os.path.join(log_path, f"{env_id}_adapter", "adapter.pth")
        adapter = Adapter(policy.observation_space, output_size=feature_encoder.mlp_output_size)
        adapter.load_state_dict(th.load(adapter_path))
        adapter.eval()
        if args.plot_encoder_output:
            predicted_extrinsics_plotter = Plotter(name="predicted_extrinsics")

    # ######################### Enjoy Loop ######################### #

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []

    try:
        for _ in range(args.n_timesteps):

            obs_np = obs
            obs = obs_as_tensor(obs_np, model.device)
            true_extrinsics = feature_encoder(obs)
            if args.plot_encoder_output:
                true_visual_extrinsics = true_extrinsics[:, -feature_encoder.visual_output_size :]
                true_extrinsics_plotter.update(true_visual_extrinsics.detach().cpu().numpy())
                heightmap_logger.update(obs_np["visual"])

            if args.adapter:
                predicted_extrinsics = adapter(obs)
                if args.plot_encoder_output:
                    predicted_visual_extrinsics = predicted_extrinsics[:, -feature_encoder.visual_output_size :]
                    predicted_extrinsics_plotter.update(predicted_visual_extrinsics.detach().cpu().numpy())

            extrinsics = predicted_extrinsics if args.adapter else true_extrinsics
            output = base_policy(extrinsics)
            action = base_policy_action(output[0])
            value = base_policy_value(output[1])
            # Clip and perform action
            clipped_action = action.detach().cpu().numpy()
            if isinstance(model.action_space, gym.spaces.Box):
                clipped_action = np.clip(clipped_action, model.action_space.low, model.action_space.high)
            obs, reward, done, infos = env.step(clipped_action)

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0
                    state = None

                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0
    except KeyboardInterrupt:
        pass

    # ######################### Print stats ######################### #

    if args.plot_encoder_output:
        true_extrinsics_plotter.plot()
        heightmap_logger.save()
        if args.adapter:
            predicted_extrinsics_plotter.plot()

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()


if __name__ == "__main__":
    main()
