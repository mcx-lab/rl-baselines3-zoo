#!/usr/bin/env python3

import rospy
import argparse
import os
import sys
from stable_baselines3 import PPO
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
    args, _ = parser.parse_known_args()

    env_id = "A1GymEnv-v0"
    log_path = args.log_path

    # ######################### Load model ######################### #

    model_path = os.path.join(log_path, f"{env_id}.zip")
    model = PPO.load(model_path, deterministic=True)
    print(f"Loaded model from {model_path}")

    # ######################### ROS node ######################### #

    global _obs

    rospy.init_node("rl_algo", anonymous=True)
    rospy.Subscriber("observations", Observation, callback_obs)
    pub_action = rospy.Publisher("actions", QuadrupedLegPos, queue_size=10)
    rate = rospy.Rate(33)  # hz
    while not rospy.is_shutdown():
        action, _ = model.predict(_obs, state=None, deterministic=True)
        # publish action
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
        # update stored observation
        _obs_lastaction = list(action)
        rate.sleep()


if __name__ == "__main__":
    main()
