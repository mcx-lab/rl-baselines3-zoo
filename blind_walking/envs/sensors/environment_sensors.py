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
from blind_walking.envs.sensors import sensor
from numpy.lib.function_base import _angle_dispatcher

_ARRAY = typing.Iterable[float]  # pylint:disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint:disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any]  # pylint:disable=invalid-name


class LastActionSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the last action taken."""

    def __init__(
        self,
        num_actions: int,
        lower_bound: _FLOAT_OR_ARRAY = -np.pi,
        upper_bound: _FLOAT_OR_ARRAY = np.pi,
        name: typing.Text = "LastAction",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
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

        super(LastActionSensor, self).__init__(
            name=name,
            shape=(self._num_actions,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.

        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the last action of the environment."""
        return self._env.last_action


class ControllerKpSensor(sensor.BoxSpaceSensor):
    """
    A sensor that reports the Kp coefficients
    used in the PD controller that converts angles to torques
    """

    def __init__(
        self,
        num_motors: int,
        lower_bound: _FLOAT_OR_ARRAY = 0,
        upper_bound: _FLOAT_OR_ARRAY = 100,
        name: typing.Text = "ControllerKp",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:

        """Constructs ControllerKpSensor.
        Args:
          lower_bound: the lower bound of the gains
          upper_bound: the upper bound of the gains
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        super(ControllerKpSensor, self).__init__(
            name=name,
            shape=(num_motors,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the Kp coefficients."""
        return self._env.robot.GetMotorPositionGains()


class ControllerKdSensor(sensor.BoxSpaceSensor):
    """
    A sensor that reports the Kd coefficients
    used in the PD controller that converts angles to torques
    """

    def __init__(
        self,
        num_motors: int,
        lower_bound: _FLOAT_OR_ARRAY = 0.0,
        upper_bound: _FLOAT_OR_ARRAY = 2.0,
        name: typing.Text = "ControllerKd",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs ControllerKdSensor.
        Args:
          lower_bound: the lower bound of the gain
          upper_bound: the upper bound of the gain
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        super(ControllerKdSensor, self).__init__(
            name=name,
            shape=(num_motors,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the Kd coefficients."""
        return self._env._robot.GetMotorVelocityGains()


class MotorStrengthSensor(sensor.BoxSpaceSensor):
    """
    A sensor that reports the relative motor strength for each joint
    """

    def __init__(
        self,
        num_motors: int,
        lower_bound: _FLOAT_OR_ARRAY = 0.0,
        upper_bound: _FLOAT_OR_ARRAY = 1.5,
        name: typing.Text = "MotorStrength",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs MotorStrengthSensor.
        Args:
          lower_bound: the lower bound of the gains
          upper_bound: the upper bound of the gains
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        super(MotorStrengthSensor, self).__init__(
            name=name,
            shape=(num_motors,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the relative motor strength (1 = full strength)."""
        return self._env._robot.GetMotorStrengthRatios()


class FootFrictionSensor(sensor.BoxSpaceSensor):
    def __init__(
        self,
        num_legs: int = 4,
        lower_bound: _FLOAT_OR_ARRAY = 0.0,
        upper_bound: _FLOAT_OR_ARRAY = 5.0,
        name: typing.Text = "FootFriction",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs FootFrictionSensor.
        Args:
          lower_bound: the lower bound of the target position
          upper_bound: the upper bound of the target position
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        super(FootFrictionSensor, self).__init__(
            name=name,
            shape=(num_legs,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the friction for each foot."""
        return self._env._robot.GetFootFriction()


class TargetPositionSensor(sensor.BoxSpaceSensor):
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
        """Constructs TargetPositionSensor.
        Args:
          lower_bound: the lower bound of the target position
          upper_bound: the upper bound of the target position
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        # Get data from file
        filepath = "blind_walking/envs/tasks/target_positions.csv"
        with open(filepath, newline="") as f:
            reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
            self._data = list(reader)

        super(TargetPositionSensor, self).__init__(
            name=name,
            shape=(2,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

        self._max_distance = max_distance
        self._distance = self._max_distance

        self._last_base_pos = np.zeros(3)
        self._current_base_pos = np.zeros(3)
        self._last_yaw = 0
        self._current_yaw = 0

    def on_step(self, env):
        self._last_base_pos = self._current_base_pos
        self._current_base_pos = self._env._robot.GetBasePosition()
        self._last_yaw = self._current_yaw
        self._current_yaw = self._env._robot.GetBaseRollPitchYaw()[2]

        # # Hardcoded, for better training of speed change
        # speed_timestep_signals = [1900, 1600, 1300, 1000]
        # target_speeds = [0.0, 0.014, 0.016, 0.018]
        # for i, t in enumerate(speed_timestep_signals):
        #     if env._env_step_counter > t:
        #         self._distance = target_speeds[i]
        #         break

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env
        self._distance = self._max_distance

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
        target_distance = np.linalg.norm([dx_target_local, dy_target_local])
        # If target is too far, scale down to maximum possible
        if target_distance and abs(target_distance) > self._distance:
            scale_ratio = self._distance / target_distance
            dx_target_local = dx_target_local * scale_ratio
            dy_target_local = dy_target_local * scale_ratio
        return [dx_target_local, dy_target_local]

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local


class LocalDistancesToGroundSensor(sensor.BoxSpaceSensor):
    """A sensor that detects the local terrain height around the robot"""

    def __init__(
        self,
        grid_unit: float = 0.05,
        grid_size: int = 16,
        lower_bound: _FLOAT_OR_ARRAY = -100,
        upper_bound: _FLOAT_OR_ARRAY = 100,
        name: typing.Text = "LocalDistancesToGround",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs LocalDistancesToGroundSensor.

        Args:
          grid_unit: Side length of one square in the grid
          grid_size: Number of squares along one side of grid
          lower_bound: the lower bound of the distance to ground.
          upper_bound: the upper bound of the distance to ground.
          name: the name of the sensor.
          dtype: data type of sensor value.
        """
        self._env = None
        self.grid_unit = grid_unit
        self.grid_size = grid_size

        super(LocalDistancesToGroundSensor, self).__init__(
            name=name,
            shape=(grid_size ** 2,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the local distances to ground"""
        return self._env.robot.GetLocalDistancesToGround(grid_unit=self.grid_unit, grid_size=self.grid_size).reshape(-1)


class LocalTerrainViewSensor(sensor.BoxSpaceSensor):
    """A sensor that gets a view of the local terrain around the robot"""

    def __init__(
        self,
        grid_unit: float = 0.1,
        grid_size: typing.Tuple[int] = (10, 10),
        transform: typing.Tuple[float] = (0, 0),
        lower_bound: _FLOAT_OR_ARRAY = -100,
        upper_bound: _FLOAT_OR_ARRAY = 100,
        name: typing.Text = "LocalTerrainView",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs LocalTerrainViewSensor.

        Args:
          grid_unit: Side length of one square in the grid
          grid_size: Number of squares along one side of grid
          lower_bound: the lower bound of the terrain view.
          upper_bound: the upper bound of the terrain view.
          name: the name of the sensor.
          dtype: data type of sensor value.
        """
        self._env = None
        self.grid_unit = grid_unit
        self.grid_size = grid_size
        self.transform = transform

        super(LocalTerrainViewSensor, self).__init__(
            name=name,
            shape=(1, grid_size[0], grid_size[1]),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the local distances to ground"""
        return self._env.robot.GetLocalTerrainView(
            grid_unit=self.grid_unit, grid_size=self.grid_size, transform=self.transform
        ).reshape(1, self.grid_size[0], self.grid_size[1])


class PhaseSensor(sensor.BoxSpaceSensor):
    """
    A sensor that returns a 2D unit vector corresponding to a point in a gait cycle
    """

    def __init__(
        self,
        init_angle: float = 0,
        frequency: float = 1.0,  # Hertz
        lower_bound: _FLOAT_OR_ARRAY = -1.0,
        upper_bound: _FLOAT_OR_ARRAY = 1.0,
        name: typing.Text = "Phase",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs PhaseSensor.
        Args:
          init_phase: Initial phase angle at env_time_step = 0
          frequency: Number of cycles per second
        """
        self._env = None
        self.init_angle = init_angle
        self.frequency = frequency

        super(PhaseSensor, self).__init__(
            name=name,
            shape=(2,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    @property
    def cycle_delta(self):
        """Return the fraction of a cycle traversed after 1 time step"""
        return self.frequency * self._env.env_time_step

    @staticmethod
    def angle_to_vector(angle):
        """Convert a 1D angle into the corresponding 2D unit vector"""
        return np.array([np.cos(angle), np.sin(angle)])

    def _get_observation(self) -> _ARRAY:
        """Returns the current phase value"""
        cycle = self._env.env_step_counter * self.cycle_delta
        # Get the angle corresponding to the cycle
        angle = cycle * 2 * np.pi + self.init_angle
        return self.angle_to_vector(angle)
