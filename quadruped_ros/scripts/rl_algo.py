#!/usr/bin/env python3

import rospy
import argparse
import os
import numpy as np
import sys
import pickle
from stable_baselines3 import PPO
from sensor_msgs.msg import Imu, JointState

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quadruped_ros.msg import (
    MotorCmds,
    BaseVelocitySensor,
)
from blind_walking.robots.action_filter import ActionFilterButter

LAIKAGO_DEFAULT_ABDUCTION_ANGLE = 0
LAIKAGO_DEFAULT_HIP_ANGLE = 0.67
LAIKAGO_DEFAULT_KNEE_ANGLE = -1.25
LAIKAGO_DEFAULT_POSE = np.array(
    [
        LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
        LAIKAGO_DEFAULT_HIP_ANGLE,
        LAIKAGO_DEFAULT_KNEE_ANGLE,
    ]
    * 4
)


def nn_action_to_motor_angle(action: np.ndarray) -> np.ndarray:
    """Converts NN action to a motor angle in radians.

    Adapted from blind_walking.env_wrappers.simple_openloop.LaikagoPoseOffsetGenerator
    """
    assert action.shape == LAIKAGO_DEFAULT_POSE.shape
    return LAIKAGO_DEFAULT_POSE + action


def build_action_filter():
    """Constructs action filter used in simulation.

    Adapted from minitaur.Minitaur._BuildActionFilter
    """
    sampling_rate = 1 / (0.001 * 30)
    num_joints = 12
    a_filter = ActionFilterButter(sampling_rate=sampling_rate, num_joints=num_joints)
    default_action = LAIKAGO_DEFAULT_POSE
    a_filter.init_history(default_action)

    return a_filter


def filter_action(action_filter: ActionFilterButter, robot_action: np.ndarray):
    """Applies action filter used in simulation.

    Adapted from minitaur.Minitaur._FilterAction
    """
    filtered_action = action_filter.filter(robot_action)
    return filtered_action


class ROSController:
    """Interface for sending and receiving messages over ROS"""

    def __init__(self):
        # Subscriber variables
        self._obs_base_velocity = [0.0] * 2
        self._obs_imu_state = [0.0] * 6
        self._obs_motor_positions = [0.0] * 12
        self._obs_motor_velocities = [0.0] * 12

        # Publisher variables
        self._motor_commands_msg = MotorCmds()

        # Initialize ROS node
        rospy.init_node("NeuralNetController", anonymous=True)

        # Set up subscribers
        rospy.Subscriber("obs_base_velocity", BaseVelocitySensor, self.on_receive_base_velocity)
        rospy.Subscriber("obs_imu_state", Imu, self.on_receive_imu_state)
        rospy.Subscriber("obs_motor_state", JointState, self.on_receive_motor_state)

        # Set up publishers
        self.motor_command_publisher = rospy.Publisher("motor_commands", MotorCmds, queue_size=3)

    def on_receive_base_velocity(self, obs):
        """Callback for updating base velocity from ROS topic"""
        self._obs_basevelocity[0] = obs.vx
        self._obs_basevelocity[1] = obs.vy

    def on_receive_imu_state(self, obs):
        """Callback for updating IMU state from ROS topic"""
        self._obs_imu_state[0] = obs.linear_acceleration.x
        self._obs_imu_state[1] = obs.linear_acceleration.y
        self._obs_imu_state[2] = obs.linear_acceleration.z

        self._obs_imu_state[3] = obs.angular_velocity.x
        self._obs_imu_state[4] = obs.angular_velocity.y
        self._obs_imu_state[5] = obs.angular_velocity.z

    def on_receive_motor_state(self, obs):
        """Callback for updating motor state from ROS topic"""
        self._obs_motor_positions = obs.position
        self._obs_motor_velocities = obs.velocity

    def publish_action(self, motor_commands: np.ndarray):
        """Callback for publishing motor commands over ROS topics"""
        self._motor_commands_msg.motorCmds = motor_commands
        self.motor_command_publisher.publish(self._motor_commands_msg)

    def get_robot_observation(self):
        return np.concatenate(
            [
                self._obs_base_velocity,
                self._obs_imu_state,
                self._obs_motor_positions,
                self._obs_motor_velocities,
            ]
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, help="Path to folder containing pre-trained model")
    parser.add_argument("--env-id", type=str, default="A1GymEnv-v0", help="ID of OpenAI gym env to lookup trained model for")
    return parser.parse_known_args()[0]


def main():  # noqa: C901

    args = get_args()

    # ######################### Load model ######################### #

    model_path = os.path.join(args.log_path, f"{args.env_id}.zip")
    model = PPO.load(model_path, deterministic=True)
    print(f"Loaded model from {model_path}")

    # Set up normalization wrapper
    vecnorm_path = os.path.join(args.log_path, args.env_id, "vecnormalize.pkl")
    with open(vecnorm_path, "rb") as f:
        vecnorm = pickle.load(f)

    # Set up action filter
    action_filter = build_action_filter()

    # ######################### ROS node ######################### #

    controller = ROSController()

    # Hardcode target_pos to be in forward direction
    target_pos = np.array([0.02, 0.0])

    rate = rospy.Rate(33)  # hz
    while not rospy.is_shutdown():
        raw_obs = np.concatenate([controller.get_robot_observation(), target_pos])
        norm_obs = vecnorm.normalize_obs(raw_obs)
        nn_action, _ = model.predict(norm_obs, deterministic=True)
        nn_action = np.clip(nn_action, -0.5, 0.5)
        motor_angles = nn_action_to_motor_angle(nn_action)
        motor_angles = filter_action(action_filter, motor_angles)
        controller.publish_action(motor_angles)
        rate.sleep()


if __name__ == "__main__":
    main()
