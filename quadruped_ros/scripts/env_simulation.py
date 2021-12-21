#!/usr/bin/env python3

import os
import sys
import rospy
import yaml
import argparse
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils.import_envs
from utils import create_test_env, get_saved_hyperparams
from quadruped_ros.msg import (
    QuadrupedLegPos,
    Observation,
)


_ctrl_actions = [0.0] * 12


def callback_action(msg_action):
    global _ctrl_actions
    _ctrl_actions[0] = msg_action.fr.hip.pos
    _ctrl_actions[1] = msg_action.fr.upper.pos
    _ctrl_actions[2] = msg_action.fr.lower.pos
    _ctrl_actions[3] = msg_action.fl.hip.pos
    _ctrl_actions[4] = msg_action.fl.upper.pos
    _ctrl_actions[5] = msg_action.fl.lower.pos
    _ctrl_actions[6] = msg_action.br.hip.pos
    _ctrl_actions[7] = msg_action.br.upper.pos
    _ctrl_actions[8] = msg_action.br.lower.pos
    _ctrl_actions[9] = msg_action.bl.hip.pos
    _ctrl_actions[10] = msg_action.bl.upper.pos
    _ctrl_actions[11] = msg_action.bl.lower.pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--log-path", help="Path to folder containing pre-trained model", type=str, default="./")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    args, _ = parser.parse_known_args()

    env_id = "A1GymEnv-v0"
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
        should_render=True,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    obs = env.reset()

    # ######################### ROS node ######################### #

    rospy.init_node('quadruped_simulator', anonymous=True)
    rospy.Subscriber("actions", QuadrupedLegPos, callback_action)
    pub_obs = rospy.Publisher("observations", Observation, queue_size=10)

    rate = rospy.Rate(33)  # Hz
    while not rospy.is_shutdown():
        # simulation step
        obs, reward, done, infos = env.step(_ctrl_actions)
        env.render("human")
        # pub observation
        obs_msg = Observation()
        obs_msg.data = obs[0]
        pub_obs.publish(obs_msg)
        # sleep rate
        rate.sleep()
