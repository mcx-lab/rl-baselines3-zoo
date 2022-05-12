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
import os

import torch as th

"""Utilities for building environments."""
from gait_imitation.envs import locomotion_gym_config, locomotion_gym_env
from gait_imitation.envs.env_modifiers import heightfield, stairs, train_course
from gait_imitation.envs.env_wrappers import observation_dictionary_split_by_encoder_wrapper as obs_split_wrapper
from gait_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from gait_imitation.envs.env_wrappers import simple_openloop, trajectory_generator_wrapper_env
from gait_imitation.envs.sensors import cpg_sensors, environment_sensors, robot_sensors, sensor_wrappers
from gait_imitation.envs.tasks import forward_task, forward_task_pos, imitation_task
from gait_imitation.envs.utilities import controllable_env_randomizer_from_config
from gait_imitation.robots import a1, laikago, robot_config


def build_regular_env(
    robot_class,
    enable_rendering=False,
    on_rack=False,
    action_limit=(0.5, 0.5, 0.5),
    robot_sensor_list=None,
    env_sensor_list=None,
    env_randomizer_list=None,
    task=None,
    # CPG sensor kwargs
    **kwargs,
):

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 10
    sim_params.enable_action_interpolation = False
    sim_params.enable_action_filter = True
    sim_params.enable_clip_motor_commands = True
    sim_params.robot_on_rack = on_rack

    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    if robot_sensor_list is None:
        robot_sensor_list = [
            sensor_wrappers.HistoricSensorWrapper(
                robot_sensors.IMUSensor(channels=["R", "P", "dR", "dP", "dY"]), num_history=3
            ),
            sensor_wrappers.HistoricSensorWrapper(robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS), num_history=3),
            environment_sensors.TargetPositionSensor(target_velocity=0.5),
            cpg_sensors.ReferenceGaitSensor(**kwargs),
        ]

    if env_randomizer_list is None:
        env_randomizer_list = []

    if task is None:
        task = imitation_task.ImitationTask()

    env = locomotion_gym_env.LocomotionGymEnv(
        gym_config=gym_config,
        robot_class=robot_class,
        robot_sensors=robot_sensor_list,
        env_sensors=env_sensor_list,
        task=task,
        env_randomizers=env_randomizer_list,
    )

    env = obs_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
        env,
        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=action_limit),
    )

    return env