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
from blind_walking.envs.sensors import sensor
from numpy.lib.function_base import _angle_dispatcher

_ARRAY = typing.Iterable[float]  # pylint:disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint:disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any]  # pylint:disable=invalid-name


class ForwardTargetPositionSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the relative target position."""

    def __init__(
        self,
        max_distance: float = 0.022,
        lower_bound: _FLOAT_OR_ARRAY = -1.0,
        upper_bound: _FLOAT_OR_ARRAY = 1.0,
        name: typing.Text = "TargetPosition",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs ForwardTargetPositionSensor.
        Args:
          lower_bound: the lower bound of the target position
          upper_bound: the upper bound of the target position
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        super(ForwardTargetPositionSensor, self).__init__(
            name=name,
            shape=(2,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

        self._max_distance = max_distance

        self._last_base_pos = np.zeros(3)
        self._current_base_pos = np.zeros(3)
        self._last_yaw = 0
        self._current_yaw = 0

    def on_step(self, env):
        self._last_base_pos = self._current_base_pos
        self._current_base_pos = self._env._robot.GetBasePosition()
        self._last_yaw = self._current_yaw
        self._current_yaw = self._env._robot.GetTrueBaseRollPitchYaw()[2]

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

        self._current_base_pos = self._env._robot.GetBasePosition()
        self._last_base_pos = self._current_base_pos
        self._current_yaw = self._env._robot.GetTrueBaseRollPitchYaw()[2]
        self._last_yaw = self._current_yaw

    def _get_observation(self) -> _ARRAY:
        # target y position is always zero
        dy_target = 0 - self._current_base_pos[1]
        # give some leeway for the robot to walk forward
        dy_target = max(min(dy_target, self._max_distance / 2), -self._max_distance / 2)
        # target x position is always forward
        dx_target = np.sqrt(pow(self._max_distance, 2) - pow(dy_target, 2))
        # Transform to local frame
        dx_target_local, dy_target_local = self.to_local_frame(dx_target, dy_target, self._current_yaw)
        return [dx_target_local, dy_target_local]

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local
