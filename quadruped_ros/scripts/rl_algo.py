#!/usr/bin/env python3

import rospy
import argparse
import os
import numpy as np
import sys
import pickle
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
from blind_walking.robots.action_filter import ActionFilterButter

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
    # ['FR_hip_joint', 'FR_upper_joint', 'FR_lower_joint',
    # 'FL_hip_joint', 'FL_upper_joint', 'FL_lower_joint', 
    # 'RR_hip_joint', 'RR_upper_joint', 'RR_lower_joint', 
    # 'RL_hip_joint', 'RL_upper_joint', 'RL_lower_joint'] 
    _obs_motors[:12] = obs.position
    _obs_motors[12:] = obs.velocity


def callback_targetpos(obs):
    global _obs_targetpos
    _obs_targetpos[0] = obs.dx
    _obs_targetpos[1] = obs.dy


def callback_heightmap(obs):
    global _obs_heightmap
    _obs_heightmap = list(obs.data)

LAIKAGO_DEFAULT_ABDUCTION_ANGLE = 0
LAIKAGO_DEFAULT_HIP_ANGLE = 0.67
LAIKAGO_DEFAULT_KNEE_ANGLE = -1.25
LAIKAGO_DEFAULT_POSE = np.array([
    LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
    LAIKAGO_DEFAULT_HIP_ANGLE,
    LAIKAGO_DEFAULT_KNEE_ANGLE,
] * 4)

def nn_action_to_motor_angle(action: np.ndarray) -> np.ndarray:
    """ Converts NN action to a motor angle in radians. 

    Adapted from blind_walking.env_wrappers.simple_openloop.LaikagoPoseOffsetGenerator
    """
    assert action.shape == LAIKAGO_DEFAULT_POSE.shape
    return LAIKAGO_DEFAULT_POSE + action

def build_action_filter():
    """ Constructs action filter used in simulation. 
    
    Adapted from minitaur.Minitaur._BuildActionFilter
    """
    sampling_rate = 1 / (0.001 * 30)
    num_joints = 12
    a_filter = ActionFilterButter(sampling_rate=sampling_rate, num_joints=num_joints)
    default_action = LAIKAGO_DEFAULT_POSE
    a_filter.init_history(default_action)

    return a_filter

def filter_action(action_filter: ActionFilterButter, robot_action: np.ndarray):
    """ Applies action filter used in simulation. 
    
    Adapted from minitaur.Minitaur._FilterAction
    """
    filtered_action = action_filter.filter(robot_action)
    return filtered_action

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

    vecnorm_path = os.path.join(log_path, env_id, "vecnormalize.pkl")
    with open(vecnorm_path, "rb") as f:
        vecnorm = pickle.load(f)

    # ######################### ROS node ######################### #

    global _obs_lastaction

    action_filter = build_action_filter()

    rospy.init_node("rl_algo", anonymous=True)
    rospy.Subscriber("obs_basevelocity", BaseVelocitySensor, callback_basevelocity)
    rospy.Subscriber("obs_imu", Imu, callback_imu)
    rospy.Subscriber("obs_motors", JointState, callback_motors)  # motor angles and velocity
    rospy.Subscriber("obs_targetpos", TargetPositionSensor, callback_targetpos)
    rospy.Subscriber("obs_heightmap", HeightmapSensor, callback_heightmap)
    pub_action = rospy.Publisher("actions", QuadrupedLegPos, queue_size=10)
    rate = rospy.Rate(33)  # hz
    while not rospy.is_shutdown():
        obs = _obs_basevelocity + _obs_imu + _obs_motors + _obs_lastaction + _obs_targetpos + _obs_heightmap
        # normalise observation
        norm_obs = vecnorm.normalize_obs(obs)
        # predict action
        nn_action, _ = model.predict(norm_obs, state=None, deterministic=True)
        nn_action = np.clip(nn_action, -0.5, 0.5)
        robot_action = nn_action_to_motor_angle(nn_action) 
        robot_action = filter_action(action_filter, robot_action)
        # publish action
        msg_action = QuadrupedLegPos()
        msg_action.fr.hip.pos = robot_action[0]
        msg_action.fr.upper.pos = robot_action[1]
        msg_action.fr.lower.pos = robot_action[2]
        msg_action.fl.hip.pos = robot_action[3]
        msg_action.fl.upper.pos = robot_action[4]
        msg_action.fl.lower.pos = robot_action[5]
        msg_action.br.hip.pos = robot_action[6]
        msg_action.br.upper.pos = robot_action[7]
        msg_action.br.lower.pos = robot_action[8]
        msg_action.bl.hip.pos = robot_action[9]
        msg_action.bl.upper.pos = robot_action[10]
        msg_action.bl.lower.pos = robot_action[11]
        pub_action.publish(msg_action)
        # update stored observation
        _obs_lastaction = list(robot_action)
        rate.sleep()


if __name__ == "__main__":
    main()
