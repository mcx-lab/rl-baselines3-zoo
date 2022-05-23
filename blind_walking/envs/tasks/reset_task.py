import math

import numpy as np

SITTING_POSE = np.array([0.0, 0.0, 0.11, 0.0, 0.0, 0.0, 1.0] + [0, 1.17752553, -2.69719727] * 4)
STANDING_POSE = np.array([0.0, 0.0, 0.25870023, 0.0, 0.0, 0.0, 1.0] + [0, 0.9, -1.8] * 4)

JOINT_WEIGHTS = np.array([1.0, 0.75, 0.5] * 4)


class ResetTask(object):
    def __init__(self, num_legs=4, num_motors=12):
        """Initializes the task."""
        self._num_legs = num_legs
        self._num_motors = num_motors
        self.env_time_step = -1

        self.current_base_pos = np.zeros(3)
        self.current_base_rpy = np.zeros(3)
        self.current_base_rpy_rate = np.zeros(3)
        self.current_motor_positions = np.zeros(num_motors)
        self.current_motor_velocities = np.zeros(num_motors)
        self.current_motor_torques = np.zeros(num_motors)
        self.current_base_orientation = np.zeros(3)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self.update(env)

    def update(self, env):
        """Updates the internal state of the task."""
        self.env_time_step = env.env_time_step
        self.current_base_pos = env.robot.GetBasePosition()
        self.current_base_rpy = env.robot.GetBaseRollPitchYaw()
        self.current_base_rpy_rate = env.robot.GetBaseRollPitchYawRate()
        self.current_motor_positions = env.robot.GetMotorAngles()
        self.current_motor_velocities = env.robot.GetMotorVelocities()
        self.current_motor_torques = env.robot.GetMotorTorques()
        self.current_base_orientation = env.robot.GetBaseOrientation()

    def done(self, env):
        """Checks if the episode is over.
        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        return rot_mat[-1] < 0.5

    def _calc_reward_stand(self):
        tar_h = STANDING_POSE[2]
        pos_size = 3
        rot_size = 4

        root_pos = self.current_base_pos
        root_h = root_pos[2]
        h_err = tar_h - root_h
        h_err /= tar_h
        h_err = np.clip(h_err, 0.0, 1.0)
        r_height = 1.0 - h_err

        tar_pose = STANDING_POSE[(pos_size + rot_size) :]
        joint_pose = self.current_motor_positions
        pose_diff = tar_pose - joint_pose
        pose_diff = JOINT_WEIGHTS * pose_diff ** 2
        pose_err = np.sum(pose_diff)
        r_pose = np.exp(-0.6 * pose_err)

        tar_vel = 0.0
        joint_vel = self.current_motor_velocities
        vel_diff = tar_vel - joint_vel
        vel_diff = vel_diff * vel_diff
        vel_err = np.sum(vel_diff)
        r_vel = np.exp(-0.02 * vel_err)

        r_stand = 0.2 * r_height + 0.6 * r_pose + 0.2 * r_vel
        return r_stand

    def reward(self, env):
        """Get the reward without side effects.

        Also return a dict of reward components"""
        del env

        weighted_objectives = {"stand_reward": self._calc_reward_stand() * 1.0}

        reward = sum([o for o in weighted_objectives.values()])
        assert self.env_time_step > 0
        reward *= self.env_time_step
        return reward, weighted_objectives
