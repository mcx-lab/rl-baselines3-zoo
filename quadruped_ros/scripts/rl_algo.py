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
    QuadrupedLeg,
    IMUSensor,
    BaseVelocitySensor,
    TargetPositionSensor,
    HeightmapSensor,
)


_obs_basevelocity = [0.0] * 2
_obs_imu = [0.0] * 6
_obs_motors = [0.0] * 24
_obs_lastaction = [0.0] * 12
_obs_targetpos = [0.0] * 2
_obs_heightmap = [0.0] * 10


def callback_basevelocity(obs):
    global _obs_basevelocity
    _obs_basevelocity[0] = obs.vx
    _obs_basevelocity[1] = obs.vy


def callback_imu(obs):
    global _obs_imu
    _obs_imu[0] = obs.linear_acceleration.x
    _obs_imu[1] = obs.linear_acceleration.y
    _obs_imu[2] = obs.linear_acceleration.z

    _obs_imu[3] = obs.angular_velocity.x
    _obs_imu[4] = obs.angular_velocity.y
    _obs_imu[5] = obs.angular_velocity.z


def callback_motors(obs):
    global _obs_motors
    _obs_motors[0] = obs.fr.hip.q
    _obs_motors[1] = obs.fr.upper.q
    _obs_motors[2] = obs.fr.lower.q
    _obs_motors[3] = obs.fl.hip.q
    _obs_motors[4] = obs.fl.upper.q
    _obs_motors[5] = obs.fl.lower.q
    _obs_motors[6] = obs.br.hip.q
    _obs_motors[7] = obs.br.upper.q
    _obs_motors[8] = obs.br.lower.q
    _obs_motors[9] = obs.bl.hip.q
    _obs_motors[10] = obs.bl.upper.q
    _obs_motors[11] = obs.bl.lower.q

    _obs_motors[12] = obs.fr.hip.dq
    _obs_motors[13] = obs.fr.upper.dq
    _obs_motors[14] = obs.fr.lower.dq
    _obs_motors[15] = obs.fl.hip.dq
    _obs_motors[16] = obs.fl.upper.dq
    _obs_motors[17] = obs.fl.lower.dq
    _obs_motors[18] = obs.br.hip.dq
    _obs_motors[19] = obs.br.upper.dq
    _obs_motors[20] = obs.br.lower.dq
    _obs_motors[21] = obs.bl.hip.dq
    _obs_motors[22] = obs.bl.upper.dq
    _obs_motors[23] = obs.bl.lower.dq


def callback_targetpos(obs):
    global _obs_targetpos
    _obs_targetpos[0] = obs.dx
    _obs_targetpos[1] = obs.dy


def callback_heightmap(obs):
    global _obs_heightmap
    _obs_heightmap = obs.data


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

    global _obs_lastaction

    rospy.init_node("rl_algo", anonymous=True)
    rospy.Subscriber("obs_basevelocity", BaseVelocitySensor, callback_basevelocity)
    rospy.Subscriber("obs_imu", Imu, callback_imu)
    rospy.Subscriber("obs_motors", QuadrupedLeg, callback_motors)  # motor angles and velocity
    rospy.Subscriber("obs_targetpos", TargetPositionSensor, callback_targetpos)
    rospy.Subscriber("obs_heightmap", HeightmapSensor, callback_heightmap)
    pub_action = rospy.Publisher("actions", QuadrupedLegPos, queue_size=10)
    rate = rospy.Rate(33)  # hz
    while not rospy.is_shutdown():
        obs = _obs_basevelocity + _obs_imu + _obs_motors + _obs_lastaction + _obs_targetpos + _obs_heightmap
        action, _ = model.predict(obs, state=None, deterministic=True)
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
