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

import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import locomotion_gym_config, locomotion_gym_env
from motion_imitation.envs.env_wrappers import default_task, imitation_task, imitation_wrapper_env
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from motion_imitation.envs.env_wrappers import simple_forward_task, simple_openloop, trajectory_generator_wrapper_env
from motion_imitation.envs.sensors import environment_sensors, robot_sensors, sensor_wrappers
from motion_imitation.envs.utilities import controllable_env_randomizer_from_config
from motion_imitation.robots import a1, laikago, robot_config


def build_imitation_env(imit_mode, motion_files, tar_frame_steps, enable_randomizer, enable_rendering):
    assert len(motion_files) > 0

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.num_action_repeat = 10
    sim_params.allow_knee_contact = True
    sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION

    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    sensors = [
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3
        ),
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
    ]

    task = imitation_task.ImitationTask(
        imit_mode=imit_mode,
        ref_motion_filenames=motion_files,
        enable_cycle_sync=True,
        tar_frame_steps=tar_frame_steps,
        ref_state_init_prob=0.9,
        warmup_time=0.25,
    )

    randomizers = []
    if enable_randomizer:
        randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
        randomizers.append(randomizer)

    env = locomotion_gym_env.LocomotionGymEnv(
        gym_config=gym_config, robot_class=a1.A1, env_randomizers=randomizers, robot_sensors=sensors, task=task
    )

    env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
        env, simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND)
    )

    env = imitation_wrapper_env.ImitationWrapperEnv(env)
    return env
