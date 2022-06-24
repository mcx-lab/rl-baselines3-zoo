# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for building environments."""
import os

import gym
from blind_walking.envs import locomotion_gym_config, locomotion_gym_env
from blind_walking.envs.env_wrappers import imitation_task
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.env_wrappers import simple_openloop, trajectory_generator_wrapper_env
from blind_walking.envs.sensors import cpg_sensors, environment_sensors, robot_sensors
from blind_walking.robots import a1, robot_config

data_path = os.path.join(os.getcwd(), "blind_walking/data")


def build_regular_env(
    robot_class,
    enable_rendering=False,
    on_rack=False,
    action_limit=(0.5, 0.5, 0.5),
):

    # Configure basic simulation parameters
    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 25
    sim_params.enable_action_interpolation = False
    sim_params.enable_action_filter = True
    sim_params.enable_clip_motor_commands = True
    sim_params.robot_on_rack = on_rack
    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    # Configure sensors, task
    robot_sensor_list = [
        # Proprioceptive sensors
        robot_sensors.BaseVelocitySensor(convert_to_local_frame=True),
        robot_sensors.IMUSensor(channels=["R", "P", "dR", "dP", "dY"]),
        robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
        robot_sensors.MotorVelocitySensor(num_motors=a1.NUM_MOTORS),
    ]
    env_sensor_list = [
        # High-level control inputs
        cpg_sensors.ReferenceGaitSensor(
            gait_names=["trot"],
            gait_frequency_upper=2.0,
            gait_frequency_lower=2.0,
            duty_factor_upper=0.5,
            duty_factor_lower=0.5,
            obs_steps_ahead=[0, 1, 2, 10, 50],
        ),
        environment_sensors.ForwardTargetPositionSensor(min_range=0.015, max_range=0.015),
    ]
    task = imitation_task.ImitationTask()

    # Initialize core gym env
    env = locomotion_gym_env.LocomotionGymEnv(
        gym_config=gym_config,
        robot_class=robot_class,
        robot_sensors=robot_sensor_list,
        env_sensors=env_sensor_list,
        task=task,
        data_path=data_path,
    )

    # Obs. wrapper for flattening dict
    env = gym.wrappers.FlattenObservation(env)

    # Act. wrapper for adding default position offsets
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
        env,
        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=action_limit),
    )

    return env
