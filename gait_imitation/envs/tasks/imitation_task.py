import math

import numpy as np


class ImitationTask(object):
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
        self.feet_contact_lost = np.zeros(num_legs)

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
        self.feet_contact_lost = env.robot._feet_contact_lost

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

        # Update relative target position
        self._target_pos = env._observations["TargetPosition_flatten"]
        # Assume gait sensor is last sensor
        ref_gait_sensor = env.all_sensors()[-1]
        self._reference_foot_contacts = ref_gait_sensor.get_current_reference_state()

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

        # Reward for following the target position.
        dx, dy, dz = np.array(self.current_base_pos) - np.array(self.last_base_pos)
        dx_local, dy_local = self.to_local_frame(dx, dy, self.last_base_rpy[2])
        dpos_robot = np.array([dx_local, dy_local])
        dpos_ref = self._target_pos

        dpos_diff = dpos_robot - dpos_ref
        dpos_err = dpos_diff.dot(dpos_diff)

        distance_reward = 100 * np.exp(-10000.0 * dpos_err) * self._env._env_time_step

        # Penalty for following the phase of the robot.
        feet_ground_time_robot = self._env.env_time_step - self.feet_air_time
        feet_ground_time_ref = self._reference_foot_contacts
        feet_ground_time_diff = feet_ground_time_robot - feet_ground_time_ref
        feet_ground_time_err = feet_ground_time_diff.dot(feet_ground_time_diff)

        ref_foot_contact_imitation_reward = 100 * np.exp(-1.0 * feet_ground_time_err) * self._env._env_time_step

        # Dictionary of:
        # - {name: reward * weight}
        # for all reward components
        weighted_objectives = {
            "distance": distance_reward * 1.0,
            "ref_foot_contact_imit": ref_foot_contact_imitation_reward * 1.0,
        }
        print(weighted_objectives)  

        reward = sum([o for o in weighted_objectives.values()])
        return reward, weighted_objectives

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local
