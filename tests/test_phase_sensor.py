from unittest import TestCase

import gym
import numpy as np
from blind_walking.envs.gym_envs import A1GymEnv
from blind_walking.envs.sensors import environment_sensors


def test_phase_sensor():
    frequency = 1.0
    env = A1GymEnv(
        env_sensor_list=[
            environment_sensors.TargetPositionSensor(),
            environment_sensors.PhaseSensor(num_motors=12, init_phase=0, frequency=frequency),
        ]
    )
    sensor = env.sensor_by_name("Phase_flatten")
    current_phase = -np.pi
    for i in range(1000):
        assert np.all(np.isclose(sensor.get_observation(), current_phase))
        current_phase += 2 * np.pi * frequency * env.env_time_step
        if current_phase >= np.pi:
            current_phase -= 2 * np.pi
        env.step(env.action_space.sample())
