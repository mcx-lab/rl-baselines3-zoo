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
import numpy as np
import typing
import csv

from blind_walking.envs.sensors import sensor

_ARRAY = typing.Iterable[float] # pylint:disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY] # pylint:disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any] # pylint:disable=invalid-name


class LastActionSensor(sensor.BoxSpaceSensor):
  """A sensor that reports the last action taken."""

  def __init__(self,
               num_actions: int,
               lower_bound: _FLOAT_OR_ARRAY = -1.0,
               upper_bound: _FLOAT_OR_ARRAY = 1.0,
               name: typing.Text = "LastAction",
               dtype: typing.Type[typing.Any] = np.float64) -> None:
    """Constructs LastActionSensor.

    Args:
      num_actions: the number of actions to read
      lower_bound: the lower bound of the actions
      upper_bound: the upper bound of the actions
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._num_actions = num_actions
    self._env = None

    super(LastActionSensor, self).__init__(name=name,
                                           shape=(self._num_actions,),
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           dtype=dtype)

  def on_reset(self, env):
    """From the callback, the sensor remembers the environment.

    Args:
      env: the environment who invokes this callback function.
    """
    self._env = env

  def _get_observation(self) -> _ARRAY:
    """Returns the last action of the environment."""
    return self._env.last_action

class ControllerKpCoefficientSensor(sensor.BoxSpaceSensor):
  """ 
  A sensor that reports the Kp and Kd coefficients 
  used in the PD controller that converts angles to torques
  """

  def __init__(self,
               num_motors: int,
               lower_bound: _FLOAT_OR_ARRAY = 45,
               upper_bound: _FLOAT_OR_ARRAY = 65,
               name: typing.Text = "ControllerKpCoefficient",
               dtype: typing.Type[typing.Any] = np.float64) -> None:
    """Constructs ControllerKpCoefficientSensor.

    Args:
      lower_bound: the lower bound of the gains
      upper_bound: the upper bound of the gains
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._env = None

    super(ControllerKpCoefficientSensor, self).__init__(name=name,
                                           shape=(num_motors,),
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           dtype=dtype)

  def on_reset(self, env):
    """From the callback, the sensor remembers the environment.

    Args:
      env: the environment who invokes this callback function.
    """
    self._env = env
 
  def _get_observation(self) -> _ARRAY:
    """Returns the Kp and Kd coefficients. """
    return self._env.robot.GetMotorPositionGains()

class ControllerKdCoefficientSensor(sensor.BoxSpaceSensor):
  """ 
  A sensor that reports the Kp and Kd coefficients 
  used in the PD controller that converts angles to torques
  """

  def __init__(self,
               num_motors: int,
               lower_bound: _FLOAT_OR_ARRAY = 0.3,
               upper_bound: _FLOAT_OR_ARRAY = 0.9,
               name: typing.Text = "ControllerKdCoefficient",
               dtype: typing.Type[typing.Any] = np.float64) -> None:
    """Constructs ControllerKdCoefficientSensor.

    Args:
      lower_bound: the lower bound of the gain
      upper_bound: the upper bound of the gain
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._env = None

    super(ControllerKdCoefficientSensor, self).__init__(name=name,
                                           shape=(num_motors,),
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           dtype=dtype)

  def on_reset(self, env):
    """From the callback, the sensor remembers the environment.

    Args:
      env: the environment who invokes this callback function.
    """
    self._env = env
 
  def _get_observation(self) -> _ARRAY:
    """Returns the Kp and Kd coefficients. """
    return self._env._robot.GetMotorVelocityGains()

class MotorStrengthRatiosSensor(sensor.BoxSpaceSensor):
  """ 
  A sensor that reports the Kp and Kd coefficients 
  used in the PD controller that converts angles to torques
  """

  def __init__(self,
               num_motors: int,
               lower_bound: _FLOAT_OR_ARRAY = 0.0,
               upper_bound: _FLOAT_OR_ARRAY = 1.0,
               name: typing.Text = "MotorStrengthRatios",
               dtype: typing.Type[typing.Any] = np.float64) -> None:
    """Constructs RobotMassSensor.

    Args:
      lower_bound: the lower bound of the gains
      upper_bound: the upper bound of the gains
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._env = None

    super(MotorStrengthRatiosSensor, self).__init__(name=name,
                                           shape=(num_motors,),
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           dtype=dtype)

  def on_reset(self, env):
    """From the callback, the sensor remembers the environment.

    Args:
      env: the environment who invokes this callback function.
    """
    self._env = env
 
  def _get_observation(self) -> _ARRAY:
    """Returns the Kp and Kd coefficients. """
    return self._env._robot.GetMotorStrengthRatios()

class TargetPositionSensor(sensor.BoxSpaceSensor):
  """A sensor that reports the relative target position."""

  def __init__(self,
               max_distance: float = 0.022,
               lower_bound: _FLOAT_OR_ARRAY = -1.0,
               upper_bound: _FLOAT_OR_ARRAY = 1.0,
               name: typing.Text = "TargetPosition",
               dtype: typing.Type[typing.Any] = np.float64) -> None:
    """Constructs TargetPositionSensor.
    Args:
      lower_bound: the lower bound of the target position
      upper_bound: the upper bound of the target position
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._env = None

    # Get data from file
    filepath = 'blind_walking/envs/env_wrappers/target_positions.csv'
    with open(filepath, newline='') as f:
      reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
      self._data = list(reader)

    super(TargetPositionSensor, self).__init__(name=name,
                                               shape=(2,),
                                               lower_bound=lower_bound,
                                               upper_bound=upper_bound,
                                               dtype=dtype)

    self._max_distance = max_distance
    self._last_base_pos = np.zeros(3)
    self._current_base_pos = np.zeros(3)
    self._last_yaw = 0
    self._current_yaw = 0

  def on_step(self, env):
    self._last_base_pos = self._current_base_pos
    self._current_base_pos = self._env._robot.GetBasePosition()
    self._last_yaw = self._current_yaw
    self._current_yaw = self._env._robot.GetBaseRollPitchYaw()[2]

  def on_reset(self, env):
    """From the callback, the sensor remembers the environment.
    Args:
      env: the environment who invokes this callback function.
    """
    self._env = env
    self._current_base_pos = self._env._robot.GetBasePosition()
    self._last_base_pos = self._current_base_pos
    self._current_yaw = self._env._robot.GetBaseRollPitchYaw()[2]
    self._last_yaw = self._current_yaw

  def _get_observation(self) -> _ARRAY:
    target_pos = self._data[self._env._env_step_counter]
    dx_target = target_pos[0] - self._current_base_pos[0]
    dy_target = target_pos[1] - self._current_base_pos[1]
    # Transform to local frame
    dx_target_local, dy_target_local = self.to_local_frame(dx_target, dy_target, self._current_yaw)
    # Scale to maximum possible
    if dx_target_local or dy_target_local:
      scale_ratio = self._max_distance / np.linalg.norm([dx_target_local, dy_target_local])
      dx_target_local = dx_target_local * scale_ratio
      dy_target_local = dy_target_local * scale_ratio
    return [dx_target_local, dy_target_local]

  @staticmethod
  def to_local_frame(dx, dy, yaw):
    # Transform the x and y direction distances to the robot's local frame
    dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
    dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
    return dx_local, dy_local