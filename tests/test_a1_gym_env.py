import unittest

import gym
import numpy as np
import torch
from motion_imitation.envs.gym_envs.a1_gym_env import A1GymEnv


class TestA1GymEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = A1GymEnv()

    @property
    def robot(self):
        """An alias to easily get the robot from A1GymEnv()

        Although the robot is only defined in LocomotionGymEnv,
        and A1GymEnv contains multiple wrappers on top of that,
        we can access it this way because gym.Wrapper behaves as if
        it has the attributes of the wrapped env.

        Reference: line 233 of https://github.com/openai/gym/blob/master/gym/core.py
        """
        return self.env.robot

    def test_controller_kp_getter_setter(self):
        self.env.reset()
        constant = 42.0
        # In minitaur.py, motor gains must be set together
        # We set the Kd to a dummy value we don't care about
        self.robot.SetMotorGains(constant, 0.234252)
        assert np.all(self.robot.GetMotorPositionGains() == constant)

    def test_controller_kd_getter_setter(self):
        self.env.reset()
        constant = 0.69
        # In minitaur.py, motor gains must be set together
        # We set the Kp to a dummy value we don't care about
        self.robot.SetMotorGains(23.23290, constant)
        assert np.all(self.robot.GetMotorVelocityGains() == constant)

    def test_unique_sensor_names(self):
        """Ensure all sensor names are unique

        If there are multiple sensors with the same name,
        some of them will not be actually used by the robot.
        And this will fail silently."""

        sensor_names = set()
        for sensor in self.env.all_sensors():
            name = sensor.get_name()
            assert name not in sensor_names
            sensor_names.add(name)
