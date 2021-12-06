import math

import numpy as np


class ForwardTask(object):
    def __init__(self, num_legs=4, num_motors=12):
        """Initializes the task."""
        self._num_legs = num_legs
        self._num_motors = num_motors

        self.current_base_pos = np.zeros(3)
        self.last_base_pos = np.zeros(3)
        self.current_base_rpy = np.zeros(3)
        self.last_base_rpy = np.zeros(3)
        self.current_base_rpy_rate = np.zeros(3)
        self.last_base_rpy_rate = np.zeros(3)
        self.current_motor_velocities = np.zeros(num_motors)
        self.last_motor_velocities = np.zeros(num_motors)
        self.current_motor_torques = np.zeros(num_motors)
        self.last_motor_torques = np.zeros(num_motors)
        self.current_base_orientation = np.zeros(3)
        self.last_base_orientation = np.zeros(3)
        self.current_foot_contacts = np.zeros(num_legs)
        self.last_foot_contacts = np.zeros(num_legs)
        self.feet_air_time = np.zeros(num_legs)
        self.current_action = np.zeros(num_motors)
        self.last_action = np.zeros(num_motors)
        self._target_pos = [0, 0]

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env

        self.last_base_pos = env.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos
        self.last_base_rpy = env.robot.GetBaseRollPitchYaw()
        self.current_base_rpy = self.last_base_rpy
        self.last_base_rpy_rate = env.robot.GetBaseRollPitchYawRate()
        self.current_base_rpy_rate = self.last_base_rpy_rate
        self.last_motor_velocities = env.robot.GetMotorVelocities()
        self.current_motor_velocities = self.last_motor_velocities
        self.last_motor_torques = env.robot.GetMotorTorques()
        self.current_motor_torques = self.last_motor_torques
        self.last_base_orientation = env.robot.GetBaseOrientation()
        self.current_base_orientation = self.last_base_orientation
        self.last_foot_contacts = env.robot.GetFootContacts()
        self.current_foot_contacts = self.last_foot_contacts
        self.feet_air_time = env.robot._feet_air_time
        self.last_action = env.last_action
        self.current_action = self.last_action

        self.motor_inertia = [i[0] for i in env.robot._motor_inertia]

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()
        self.last_base_rpy = self.current_base_rpy
        self.current_base_rpy = env.robot.GetBaseRollPitchYaw()
        self.last_base_rpy_rate = self.current_base_rpy_rate
        self.current_base_rpy_rate = env.robot.GetBaseRollPitchYawRate()
        self.last_motor_velocities = self.current_motor_velocities
        self.current_motor_velocities = env.robot.GetMotorVelocities()
        self.last_motor_torques = self.current_motor_torques
        self.current_motor_torques = env.robot.GetMotorTorques()
        self.last_base_orientation = self.current_base_orientation
        self.current_base_orientation = env.robot.GetBaseOrientation()
        self.last_foot_contacts = self.current_foot_contacts
        self.current_foot_contacts = env.robot.GetFootContacts()
        self.feet_air_time = env.robot._feet_air_time
        self.last_action = self.current_action
        self.current_action = env.last_action
        # Update relative target position
        self._target_pos = env._observations["TargetPosition_flatten"]

    def done(self, env):
        """Checks if the episode is over.
        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        return rot_mat[-1] < 0.5

    def reward(self, env):
        """Get the reward without side effects.

        Also return a dict of reward components"""
        del env

        alpha = 1e-2

        dx, dy, dz = np.array(self.current_base_pos) - np.array(self.last_base_pos)
        dx_local, dy_local = self.to_local_frame(dx, dy, self.last_base_rpy[2])
        dxy_local = np.array([dx_local, dy_local])
        # Reward distance travelled in target direction.
        distance_target = np.linalg.norm(self._target_pos)
        if distance_target:
            distance_towards = np.dot(dxy_local, self._target_pos) / distance_target
            distance_reward = min(distance_towards / distance_target, 1)
        else:
            distance_reward = -np.linalg.norm(dxy_local)

        # Penalty for upward translation.
        dz_reward = -np.sum(dz ** 2)

        # Penalty for roll and yaw velocity
        roll_yaw_vel_reward = -np.sum(self.current_base_rpy_rate[:2] ** 2)

        # Penalty for joint motion
        current_motor_acceleration = (self.current_motor_velocities - self.last_motor_velocities) / self._env.env_time_step
        joint_motion_reward = -np.sum(self.current_motor_velocities ** 2) - np.sum(current_motor_acceleration ** 2)

        # Penalty for high joint torques
        joint_torque_reward = -np.sum(self.current_motor_torques ** 2)

        # Penalty for action rate
        action_rate_reward = -np.sum((self.current_action - self.last_action) ** 2)

        # Reward for feet air time
        airtime_reward = np.sum(self.feet_air_time - (0.5 * self._env._env_time_step))

        # Dictionary of:
        # - {name: reward * weight}
        # for all reward components
        weighted_objectives = {
            "distance": distance_reward * 1 * self._env.env_time_step,
            "dz": dz_reward * 4 * self._env.env_time_step,
            "roll_yaw_vel": roll_yaw_vel_reward * 0.05 * self._env.env_time_step,
            "joint_motion": joint_motion_reward * 0.001 * self._env.env_time_step,
            "joint_torque": joint_torque_reward * 0.00002 * self._env.env_time_step,
            "action_rate": action_rate_reward * 0.25 * self._env.env_time_step,
            "airtime": airtime_reward * 2 * self._env.env_time_step,
        }

        reward = sum([o for o in weighted_objectives.values()])
        return reward, weighted_objectives

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local
