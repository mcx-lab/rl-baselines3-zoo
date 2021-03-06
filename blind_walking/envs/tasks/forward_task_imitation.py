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
        self.feet_contact_lost = np.zeros(num_legs)
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
        self.feet_contact_lost = env.robot._feet_contact_lost
        self.last_action = env.robot._last_action

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
        self.feet_contact_lost = env.robot._feet_contact_lost
        self.last_action = env.robot._last_action

        # Update relative target position
        self._target_pos = env._observations["TargetPosition_flatten"]
        self._reference_foot_contacts = env._observations["ReferenceGait_flatten"]

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
        distance_reward = distance_reward * self._env._env_time_step
        # Reward closeness to target position.
        dxy_err = np.linalg.norm(self._target_pos - dxy_local, 2)
        dxy_var = 1.0 * self._env._env_time_step
        dxy_reward = math.exp(math.log(alpha) * (dxy_err / dxy_var) ** 2)
        # Penalty for upward translation.
        dz_reward = -abs(dz)

        # Penalty for sideways rotation of the body.
        orientation = self.current_base_orientation
        rot_matrix = self._env.pybullet_client.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec))) * self._env._env_time_step
        # Penalty for energy usage.
        energy_reward = -np.abs(np.dot(self.current_motor_torques, self.current_motor_velocities)) * self._env._env_time_step
        energy_rot_reward = (
            -np.dot(self.motor_inertia, np.square(self.current_motor_velocities)) * self._env._env_time_step * 0.5
        )

        # Penalty for lost of more than two foot contacts
        # contact_reward = min(sum(self.current_foot_contacts), 2) - 2
        contact_reward = -self.feet_contact_lost

        # Reward for feet air time
        airtime_reward = np.sum(self.feet_air_time - (0.5 * self._env._env_time_step))

        # Penalty for action rate
        action_reward = -np.power(np.linalg.norm(self.last_action), 2) * self._env._env_time_step

        # Reward for following the phase of the robot.
        feet_ground_time = self._env.env_time_step - self.feet_air_time
        ref_foot_contact_imitation_reward = np.dot(feet_ground_time, self._reference_foot_contacts)

        # Dictionary of:
        # - {name: reward * weight}
        # for all reward components
        weighted_objectives = {
            "distance": distance_reward * 1.0,
            "dxy": dxy_reward * 0.0,
            "dz": dz_reward * 0.0,
            "shake": shake_reward * 1.5,
            "energy": energy_reward * 0.0001,
            "energy_rot": energy_rot_reward * 0.0,
            # "contact": contact_reward * 0.5,
            # "airtime": airtime_reward * 0.5,
            "action": action_reward * 0.0,
            "ref_foot_contact_imit": ref_foot_contact_imitation_reward * 0.5,
        }

        reward = sum([o for o in weighted_objectives.values()])
        return reward, weighted_objectives

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local
