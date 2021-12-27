#!/usr/bin/env python3

import os
import sys
import rospy
import yaml
import argparse
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# import utils.import_envs
# from utils import create_test_env, get_saved_hyperparams
from quadruped_ros.msg import (
    QuadrupedLegPos,
    Observation,
)


_obs = [0.0] * 92


def callback_obs(obs):
    global _obs
    _obs = list(obs.data)


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--log-path", help="Path to folder containing pre-trained model", type=str, default="./")
    # parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    args, _ = parser.parse_known_args()

    env_id = "A1GymEnv-v0"
    log_path = args.log_path

    # # ######################### Create environment ######################### #

    # set_random_seed(args.seed)
    # stats_path = os.path.join(log_path, env_id)
    # hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    # # load env_kwargs if existing
    # env_kwargs = {}
    # args_path = os.path.join(log_path, env_id, "args.yml")
    # if os.path.isfile(args_path):
    #     with open(args_path, "r") as f:
    #         loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
    #         if loaded_args["env_kwargs"] is not None:
    #             env_kwargs = loaded_args["env_kwargs"]

    # env = create_test_env(
    #     env_id,
    #     n_envs=1,
    #     stats_path=stats_path,
    #     seed=args.seed,
    #     log_dir=None,
    #     should_render=True,
    #     hyperparams=hyperparams,
    #     env_kwargs=env_kwargs,
    # )
    # obs = env.reset()

    # ######################### Load model ######################### #

    model_path = os.path.join(log_path, f"{env_id}.zip")
    model = PPO.load(model_path, deterministic=True)
    print(f"Loaded model from {model_path}")

    # ######################### ROS node ######################### #

    global _obs

    rospy.init_node("rl_algo", anonymous=True)
    rospy.Subscriber("observations", Observation, callback_obs)
    pub_action = rospy.Publisher("actions", QuadrupedLegPos, queue_size=1)
    rate = rospy.Rate(33)  # hz
    while not rospy.is_shutdown():
        action, _ = model.predict(_obs, state=None, deterministic=True)
        # obs, reward, done, infos = env.step(action)
        # env.render("human")
        # publish action
        action = action[0]
        msg_action = QuadrupedLegPos()
        msg_action.fr.hip.pos = action[0]
        msg_action.fr.upper.pos = action[1]
        msg_action.fr.lower.pos = action[2]
        msg_action.fl.hip.pos = action[3]
        msg_action.fl.upper.pos = action[4]
        msg_action.fl.lower.pos = action[5]
        msg_action.br.hip.pos = action[6]
        msg_action.br.upper.pos = action[7]
        msg_action.br.lower.pos = action[8]
        msg_action.bl.hip.pos = action[9]
        msg_action.bl.upper.pos = action[10]
        msg_action.bl.lower.pos = action[11]
        pub_action.publish(msg_action)
        rate.sleep()


if __name__ == "__main__":
    main()
