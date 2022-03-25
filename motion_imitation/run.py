# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation
from pathlib import Path

from stable_baselines.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True


def set_rand_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    seed += 97 * MPI.COMM_WORLD.Get_rank()

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return


def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
    policy_kwargs = {"net_arch": [{"pi": [512, 256], "vf": [512, 256]}], "act_fun": tf.nn.relu}

    timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
    optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

    model = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=env,
        gamma=0.95,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2,
        optim_epochs=1,
        optim_stepsize=1e-5,
        optim_batchsize=optim_batchsize,
        lam=0.95,
        adam_epsilon=1e-5,
        schedule="constant",
        policy_kwargs=policy_kwargs,
        tensorboard_log=output_dir,
        verbose=1,
    )
    return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
    if output_dir == "":
        save_path = None
    else:
        save_path = os.path.join(output_dir, "model.zip")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if output_dir != "":
        if int_save_freq > 0:
            int_dir = os.path.join(output_dir, "intermedate")
            callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir, name_prefix="model"))

    model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks)

    return


class Logger:
    def __init__(self, name: str = "log"):
        self.data = []
        self.name = name + ".csv"

    def update(self, data: np.ndarray):
        self.data.append(np.squeeze(data))

    def save(self, savedir: str = None):
        all_data = np.stack(self.data, axis=0)
        np.savetxt(os.path.join(savedir, self.name), all_data, delimiter=",")

    def clear(self):
        self.data = []


class NNActionLogger:
    def __init__(self, savedir: str):
        self.action_logger = Logger("nn_actions")
        self.savedir = savedir

    def on_step(self, actions: np.ndarray = None, **kwargs):
        self.action_logger.update(actions)

    def on_episode_end(self, **kwargs):
        self.action_logger.save(str(self.savedir))
        self.action_logger.clear()


class NNObservationLogger:
    def __init__(self, savedir: str):
        self.observation_logger = Logger("nn_observations")
        self.savedir = savedir

    def on_step(self, observations: np.ndarray = None, **kwargs):
        self.observation_logger.update(observations)

    def on_episode_end(self, **kwargs):
        self.observation_logger.save(str(self.savedir))
        self.observation_logger.clear()


class RobotStateLogger:

    log_names = ("motor_position", "motor_velocity", "motor_torque", "time")

    def __init__(self, savedir: str):
        self.loggers = {ln: Logger(ln) for ln in self.log_names}
        self.savedir = savedir

    def on_step(self, robot=None, **kwargs):
        self.loggers["motor_position"].update(robot.GetTrueMotorAngles())
        self.loggers["motor_velocity"].update(robot.GetTrueMotorVelocities())
        self.loggers["motor_torque"].update(robot.GetTrueMotorTorques())
        self.loggers["time"].update(robot.GetTimeSinceReset())

    def on_episode_end(self, **kwargs):
        for logger in self.loggers.values():
            logger.save(str(self.savedir))
            logger.clear()


def test(model, env, num_procs, num_episodes=None, callbacks=[]):

    curr_return = 0
    sum_return = 0
    episode_count = 0

    if num_episodes is not None:
        num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
    else:
        num_local_episodes = np.inf

    o = env.reset()
    while episode_count < num_local_episodes:
        a, _ = model.predict(o, deterministic=True)
        o, r, done, info = env.step(a)
        for callback in callbacks:
            callback.on_step(actions=a, observations=o, robot=env.robot)
        curr_return += r

        if done:
            o = env.reset()
            for callback in callbacks:
                callback.on_episode_end()
            sum_return += curr_return
            episode_count += 1

    sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
    episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

    mean_return = sum_return / episode_count

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Mean Return: " + str(mean_return))
        print("Episode Count: " + str(episode_count))

    return


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument(
        "--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt"
    )
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=1)
    arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
    arg_parser.add_argument(
        "--int_save_freq", dest="int_save_freq", type=int, default=0
    )  # save intermediate model every n policy steps

    args = arg_parser.parse_args()

    num_procs = MPI.COMM_WORLD.Get_size()
    print(f"Num processes: {num_procs}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")
    env = env_builder.build_imitation_env(
        motion_files=[args.motion_file],
        num_parallel_envs=num_procs,
        mode=args.mode,
        enable_randomizer=enable_env_rand,
        enable_rendering=args.visualize,
    )

    model = build_model(
        env=env,
        num_procs=num_procs,
        timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
        optim_batchsize=OPTIM_BATCHSIZE,
        output_dir=args.output_dir,
    )

    if args.model_file != "":
        model.load_parameters(args.model_file)

    stats_dir = Path("./stats").absolute()
    stats_dir.mkdir(exist_ok=True, parents=True)

    callbacks = [
        NNActionLogger(savedir=stats_dir),
        NNObservationLogger(savedir=stats_dir),
        RobotStateLogger(savedir=stats_dir),
    ]

    if args.mode == "train":
        train(
            model=model,
            env=env,
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            int_save_freq=args.int_save_freq,
        )
    elif args.mode == "test":
        test(model=model, env=env, num_procs=num_procs, num_episodes=args.num_test_episodes, callbacks=callbacks)
    else:
        assert False, "Unsupported mode: " + args.mode

    return


if __name__ == "__main__":
    main()
