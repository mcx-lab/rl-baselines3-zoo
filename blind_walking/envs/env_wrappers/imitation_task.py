import math

import numpy as np
import pybullet as p


def to_local_frame(dx, dy, yaw):
    # Transform the x and y direction distances to the robot's local frame
    dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
    dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
    return dx_local, dy_local


class ImitationTask(object):
    def __init__(self):
        """Initializes the task."""
        pass

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self.current_base_pos = env.robot.GetBasePosition()
        self.current_base_rpy = env.robot.GetBaseRollPitchYaw()
        self.update(env)

    def update(self, env):
        """Updates the internal state of the task."""
        self._reference_displacement = self._get_reference_displacement(env)
        self._actual_displacement = self._get_actual_displacement(env)
        self._reference_foot_contacts = self._get_reference_foot_contact(env)
        self._actual_foot_contacts = self._get_actual_foot_contact(env)

        self.motor_torques = env.robot.GetMotorTorques()
        self.motor_velocities = env.robot.GetMotorVelocities()
        self.base_orientation = env.robot.GetBaseOrientation()
        self.feet_air_time = env.robot._feet_air_time

        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()
        self.last_base_rpy = self.current_base_rpy
        self.current_base_rpy = env.robot.GetBaseRollPitchYaw()

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

        displacement_reward = self._calc_reward_displacement()
        shake_reward = self._calc_reward_shake()
        energy_reward = self._calc_reward_energy()
        imitation_reward = self._calc_reward_imitation()

        # Dictionary of:
        # - {name: reward * weight}
        # for all reward components
        weighted_objectives = {
            "distance": displacement_reward * 1.0,
            "shake": shake_reward * 5.0,
            "energy": energy_reward * 0.0001,
            "imitation": imitation_reward * 2.0,
        }

        reward = sum([o for o in weighted_objectives.values()])
        reward = reward * self._env.env_time_step
        return reward, weighted_objectives

    ##############################################
    # Helper functions to get control references #
    ##############################################

    def _get_reference_displacement(self, env):
        return env._observations["TargetPosition_flatten"]

    def _get_actual_displacement(self, env):
        del env
        dx, dy, _ = np.array(self.current_base_pos) - np.array(self.last_base_pos)
        dx_local, dy_local = self.to_local_frame(dx, dy, self.last_base_rpy[2])
        self._actual_displacement = np.array([dx_local, dy_local])

    def _get_reference_foot_contact(self, env):
        # Assume fixed position of ref. gait sensor
        ref_gait_sensor = env.all_sensors()[-2]
        self._reference_foot_contacts = ref_gait_sensor.get_current_reference_state()

    def _get_actual_foot_contact(self, env):
        t = env.env_time_step
        del env
        self._actual_foot_contacts = (t - 2 * self.feet_air_time) / t

    ########################################
    # Helper functions to calculate reward #
    ########################################

    def _calc_reward_displacement(self):
        """Reward term for travelling in the indicated direction"""
        # Reward distance travelled in target direction.
        displacement_diff = self._reference_displacement - self._actual_displacement
        displacement_err = displacement_diff.dot(displacement_diff)
        displacement_rew = np.exp(-4 * displacement_err)
        return displacement_rew

    def _calc_reward_shake(self):
        """Reward term for staying upright"""
        orientation = self.base_orientation
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        return shake_reward

    def _calc_reward_energy(self):
        energy_reward = -np.abs(np.dot(self.motor_torque, self.motor_velocity))
        return energy_reward

    def _calc_reward_imitation(self):
        feet_ground_time = (self._env.env_time_step - self.feet_air_time) / self._env.env_time_step
        ref_foot_contact_imitation_reward = np.dot(feet_ground_time, self._reference_foot_contacts)
        # Rescale from [-4,4] to [0,1]
        ref_foot_contact_imitation_reward += 4
        ref_foot_contact_imitation_reward /= 8
        return ref_foot_contact_imitation_reward
