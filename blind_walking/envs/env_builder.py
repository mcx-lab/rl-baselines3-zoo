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
from blind_walking.envs import locomotion_gym_config, locomotion_gym_env
from blind_walking.envs.env_modifiers import heightfield, stairs, train_course
from blind_walking.envs.env_wrappers import observation_dictionary_split_by_encoder_wrapper as obs_split_wrapper
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.env_wrappers import simple_openloop, trajectory_generator_wrapper_env
from blind_walking.envs.sensors import environment_sensors, robot_sensors
from blind_walking.envs.tasks import forward_task, forward_task_pos
from blind_walking.envs.utilities.controllable_env_randomizer_from_config import ControllableEnvRandomizerFromConfig
from blind_walking.robots import a1, laikago, robot_config


def build_regular_env(
    robot_class,
    motor_control_mode,
    enable_rendering=False,
    on_rack=False,
    action_limit=(0.75, 0.75, 0.75),
    wrap_trajectory_generator=True,
    robot_sensor_list=None,
    env_sensor_list=None,
    env_randomizer_list=None,
    env_modifier_list=None,
    task=None,
    obs_wrapper=None,
):

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.motor_control_mode = motor_control_mode
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 30
    sim_params.enable_action_interpolation = False
    sim_params.enable_action_filter = True
    sim_params.enable_clip_motor_commands = True
    sim_params.robot_on_rack = on_rack

    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    if robot_sensor_list is None:
        robot_sensor_list = [
            robot_sensors.BaseVelocitySensor(convert_to_local_frame=True, exclude_z=True),
            robot_sensors.IMUSensor(channels=["R", "P", "Y", "dR", "dP", "dY"]),
            robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
            robot_sensors.MotorVelocitySensor(num_motors=a1.NUM_MOTORS),
        ]
    if env_sensor_list is None:
        env_sensor_list = [
            environment_sensors.LastActionSensor(num_actions=a1.NUM_MOTORS),
            environment_sensors.ForwardTargetPositionSensor(max_distance=0.02),
        ]

    if env_randomizer_list is None:
        # env_randomizer_list = [ControllableEnvRandomizerFromConfig("train_params", step_sample_prob=0.004)]
        env_randomizer_list = []

    if env_modifier_list is None:
        env_modifier_list = [train_course.TrainUneven()]

    if task is None:
        task = forward_task_pos.ForwardTask()

    if obs_wrapper is None:
        obs_wrapper = obs_array_wrapper.ObservationDictionaryToArrayWrapper

    env = locomotion_gym_env.LocomotionGymEnv(
        gym_config=gym_config,
        robot_class=robot_class,
        robot_sensors=robot_sensor_list,
        env_sensors=env_sensor_list,
        task=task,
        env_randomizers=env_randomizer_list,
        env_modifiers=env_modifier_list,
    )

    env = obs_wrapper(env)
    if (motor_control_mode == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
        if robot_class == laikago.Laikago:
            env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
                env,
                trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=action_limit),
            )
        elif robot_class == a1.A1:
            env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
                env,
                trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=action_limit),
            )
    return env
