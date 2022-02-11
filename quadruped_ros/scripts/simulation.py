#!/usr/bin/env python3

import os

# import tf
import sys
import rospy
import rospkg
import threading
import pybullet as p
import pybullet_data
import pyquaternion
import numpy as np
from sensor_msgs.msg import Imu, JointState

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quadruped_ros.msg import (
    MotorCmds,
    BaseVelocitySensor,
)


class WalkingSimulation:

    # TODO
    MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
    ACTION_REPEAT = 30
    MOTOR_ID_LIST = []
    DEFAULT_POSE = []

    def __init__(self):
        self._motor_commands = [0.0] * 12

        self.initialize_ros()
        self.initialize_pybullet()

    def on_receive_motor_commands(self, msg):
        """Callback for updating motor commands from ROS topic"""
        self._motor_commands = msg.motorCmds

    def initialize_ros(self):

        # Initialize ros node
        rospy.init_node("QuadrupedSimulator", anonymous=True)

        # Set up ROS publishers
        self.base_velocity_publisher = rospy.Publisher("obs_base_velocity", BaseVelocitySensor, queue_size=30)
        self.imu_state_publisher = rospy.Publisher("obs_imu_state", Imu, queue_size=30)
        self.motor_state_publisher = rospy.Publisher("obs_motor_state", JointState, queue_size=30)

        # Set up ROS subscriber
        rospy.Subscriber("motor_cmds", MotorCmds, self.on_receive_motor_commands)


if __name__ == "__main__":

    walking_simulation = WalkingSimulation()
    walking_simulation.run()
