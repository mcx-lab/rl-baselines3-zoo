import numpy as np
import math


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

        # Update relative target position
        self._target_pos = env._observations['TargetPosition']

    def done(self, env):
        """Checks if the episode is over.
        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        return rot_mat[-1] < 0.85

    def reward(self, env):
        """Get the reward without side effects."""
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
            distance_reward = - np.linalg.norm(dxy_local)
        # Reward closeness to target position.
        dxy_err = np.linalg.norm(self._target_pos - dxy_local, 2)
        dxy_reward = math.exp(math.log(alpha)*(dxy_err/0.01)**2)
        # Penalty for upward translation.
        dz_reward = -abs(dz)

        # Penalty for sideways rotation of the body.
        orientation = self.current_base_orientation
        rot_matrix = self._env.pybullet_client.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        # Penalty for energy usage.
        energy_reward = -np.abs(
            np.dot(self.current_motor_torques,
                   self.current_motor_velocities)) * self._env._sim_time_step
        energy_rot_reward = -np.dot(self.motor_inertia, np.square(self.current_motor_velocities)) \
            * self._env._sim_time_step * 0.5

        # Penalty for lost of more than two foot contacts
        contact_reward = min(sum(self.current_foot_contacts), 2) - 2

        objectives = [distance_reward, dxy_reward, dz_reward,
                      shake_reward, energy_reward, energy_rot_reward, contact_reward]
        objective_weights = [0.01, 0.01, 0.001,
                             0.001, 0.005, 0.005, 0.0]
        weighted_objectives = [o * w for o, w in zip(objectives, objective_weights)]
        reward = sum(weighted_objectives)
        return reward

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local
