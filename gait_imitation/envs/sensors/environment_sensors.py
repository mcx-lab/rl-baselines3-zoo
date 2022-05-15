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

"""Simple sensors related to the environment."""
import csv
import typing

import numpy as np
import torch
from gait_imitation.envs.sensors import sensor
from numpy.lib.function_base import _angle_dispatcher

_ARRAY = typing.Iterable[float]  # pylint:disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint:disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any]  # pylint:disable=invalid-name


class TargetPositionSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the next target position in the robot frame."""

    def __init__(
        self,
        target_velocity: float = 1.0, # m / s
        lower_bound: _FLOAT_OR_ARRAY = -1.0,
        upper_bound: _FLOAT_OR_ARRAY = 1.0,
        name: typing.Text = "TargetPosition",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs TargetPositionSensor.
        Args:
          lower_bound: the lower bound of the target position
          upper_bound: the upper bound of the target position
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        super(TargetPositionSensor, self).__init__(
            name=name,
            shape=(2,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

        self._target_velocity = target_velocity

    def on_step(self, env):
        self._current_base_pos = env._robot.GetBasePosition()
        self._current_yaw = env._robot.GetTrueBaseRollPitchYaw()[2]

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

        # Randomize target velocity
        self._target_velocity = np.random.uniform(0.5, 1.0)
        self.on_step(env)

    def _calc_direction_unit_vector(self):
        """ Return a unit vector for the direction robot should move in """
        # target y position is always zero
        y_target = 0 
        y_actual = self._current_base_pos[1]
        # dy_target is clipped to (-0.5, 0.5) to allow robot to walk forward
        dy_target = np.clip(y_target - y_actual, -0.5, 0.5)
        # Compute dx_target
        dx_target = np.sqrt(1 - dy_target ** 2)
        # Transform to local frame
        dx_target_local, dy_target_local = self.to_local_frame(dx_target, dy_target, self._current_yaw)
        direction_vector = np.array([dx_target_local, dy_target_local])
        return direction_vector / np.linalg.norm(direction_vector)

    def _get_observation(self) -> _ARRAY:
        direction_vector = self._calc_direction_unit_vector()
        velocity_vector = self._target_velocity * direction_vector
        displacement_offset = self._env.env_time_step * velocity_vector
        return displacement_offset

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local
