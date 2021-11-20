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

"""Simple sensors related to the robot."""
import typing

import numpy as np
from blind_walking.envs.sensors import sensor
from blind_walking.robots import action_filter, minitaur_pose_utils

_ARRAY = typing.Iterable[float]  # pylint: disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint: disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any]  # pylint: disable=invalid-name


class MotorAngleSensor(sensor.BoxSpaceSensor):
    """A sensor that reads motor angles from the robot."""

    def __init__(
        self,
        num_motors: int,
        noisy_reading: bool = True,
        observe_sine_cosine: bool = False,
        lower_bound: _FLOAT_OR_ARRAY = -np.pi,
        upper_bound: _FLOAT_OR_ARRAY = np.pi,
        name: typing.Text = "MotorAngle",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs MotorAngleSensor.

        Args:
          num_motors: the number of motors in the robot
          noisy_reading: whether values are true observations
          observe_sine_cosine: whether to convert readings to sine/cosine values for
            continuity
          lower_bound: the lower bound of the motor angle
          upper_bound: the upper bound of the motor angle
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_motors = num_motors
        self._noisy_reading = noisy_reading
        self._observe_sine_cosine = observe_sine_cosine

        if observe_sine_cosine:
            super(MotorAngleSensor, self).__init__(
                name=name,
                shape=(self._num_motors * 2,),
                lower_bound=-np.ones(self._num_motors * 2),
                upper_bound=np.ones(self._num_motors * 2),
                dtype=dtype,
            )
        else:
            super(MotorAngleSensor, self).__init__(
                name=name,
                shape=(self._num_motors,),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dtype=dtype,
            )

    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_angles = self._robot.GetMotorAngles()
        else:
            motor_angles = self._robot.GetTrueMotorAngles()

        if self._observe_sine_cosine:
            return np.hstack((np.cos(motor_angles), np.sin(motor_angles)))
        else:
            return motor_angles


class MotorVelocitySensor(sensor.BoxSpaceSensor):
    """A sensor that reads motor velocities from the robot."""

    def __init__(
        self,
        num_motors: int,
        noisy_reading: bool = True,
        lower_bound: _FLOAT_OR_ARRAY = -10,
        upper_bound: _FLOAT_OR_ARRAY = 10,
        name: typing.Text = "MotorVelocity",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs MotorVelocitySensor.

        Args:
          num_motors: the number of motors in the robot
          noisy_reading: whether values are true observations
          lower_bound: the lower bound of the motor velocity
          upper_bound: the upper bound of the motor velocity
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_motors = num_motors
        self._noisy_reading = noisy_reading
        super(MotorVelocitySensor, self).__init__(
            name=name,
            shape=(self._num_motors,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_velocities = self._robot.GetMotorVelocities()
        else:
            motor_velocities = self._robot.GetTrueMotorVelocities()
        return motor_velocities


class UltrasonicSensor(sensor.BoxSpaceSensor):
    """A sensor that reads the distance between the robot feet and the ground directly below"""

    pass


class MotorTorqueSensor(sensor.BoxSpaceSensor):
    """A sensor that reads motor torques from the robot."""

    def __init__(
        self,
        num_motors: int,
        noisy_reading: bool = True,
        lower_bound: _FLOAT_OR_ARRAY = -100,
        upper_bound: _FLOAT_OR_ARRAY = 100,
        name: typing.Text = "MotorTorque",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs MotorTorqueSensor.

        Args:
          num_motors: the number of motors in the robot
          noisy_reading: whether values are true observations
          lower_bound: the lower bound of the motor torque
          upper_bound: the upper bound of the motor torque
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_motors = num_motors
        self._noisy_reading = noisy_reading
        super(MotorTorqueSensor, self).__init__(
            name=name,
            shape=(self._num_motors,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_torques = self._robot.GetMotorTorques()
        else:
            motor_torques = self._robot.GetTrueMotorTorques()
        return motor_torques


class MinitaurLegPoseSensor(sensor.BoxSpaceSensor):
    """A sensor that reads leg_pose from the Minitaur robot."""

    def __init__(
        self,
        num_motors: int,
        noisy_reading: bool = True,
        observe_sine_cosine: bool = False,
        lower_bound: _FLOAT_OR_ARRAY = -np.pi,
        upper_bound: _FLOAT_OR_ARRAY = np.pi,
        name: typing.Text = "MinitaurLegPose",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs MinitaurLegPoseSensor.

        Args:
          num_motors: the number of motors in the robot
          noisy_reading: whether values are true observations
          observe_sine_cosine: whether to convert readings to sine/cosine values for
            continuity
          lower_bound: the lower bound of the motor angle
          upper_bound: the upper bound of the motor angle
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_motors = num_motors
        self._noisy_reading = noisy_reading
        self._observe_sine_cosine = observe_sine_cosine

        if observe_sine_cosine:
            super(MinitaurLegPoseSensor, self).__init__(
                name=name,
                shape=(self._num_motors * 2,),
                lower_bound=-np.ones(self._num_motors * 2),
                upper_bound=np.ones(self._num_motors * 2),
                dtype=dtype,
            )
        else:
            super(MinitaurLegPoseSensor, self).__init__(
                name=name,
                shape=(self._num_motors,),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dtype=dtype,
            )

    def _get_observation(self) -> _ARRAY:
        motor_angles = self._robot.GetMotorAngles() if self._noisy_reading else self._robot.GetTrueMotorAngles()
        leg_pose = minitaur_pose_utils.motor_angles_to_leg_pose(motor_angles)
        if self._observe_sine_cosine:
            return np.hstack((np.cos(leg_pose), np.sin(leg_pose)))
        else:
            return leg_pose


class BaseVelocitySensor(sensor.BoxSpaceSensor):
    """A sensor that reads the robot's base velocity."""

    def __init__(
        self,
        lower_bound: _FLOAT_OR_ARRAY = -100,
        upper_bound: _FLOAT_OR_ARRAY = 100,
        convert_to_local_frame: bool = False,
        exclude_z: bool = False,
        name: typing.Text = "BaseVelocity",
        dtype: typing.Type[typing.Any] = float,
    ) -> None:
        """Constructs BaseVelocitySensor.

        Args:
          lower_bound: the lower bound of the motor velocity
          upper_bound: the upper bound of the motor velocity
          convert_to_local_frame: whether to project dx, dy to local frame based on
            robot's current yaw angle. (Note that it's a projection onto 2D plane,
            and the roll, pitch of the robot is not considered.)
          name: the name of the sensor
          dtype: data type of sensor value
        """
        size = 2 if exclude_z else 3
        super(BaseVelocitySensor, self).__init__(
            name=name,
            shape=(size,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

        self._convert_to_local_frame = convert_to_local_frame
        self._exclude_z = exclude_z

        self._last_yaw = 0
        self._last_base_velocity = np.zeros(3)
        self._current_yaw = 0
        self._current_base_velocity = np.zeros(3)

    def _get_observation(self) -> _ARRAY:
        vx, vy, vz = self._current_base_velocity
        if self._convert_to_local_frame:
            vx_local = np.cos(self._last_yaw) * vx + np.sin(self._last_yaw) * vy
            vy_local = -np.sin(self._last_yaw) * vx + np.cos(self._last_yaw) * vy
            if self._exclude_z:
                return np.array([vx_local, vy_local])
            return np.array([vx_local, vy_local, vz])

        if self._exclude_z:
            return np.array([vx, vy])
        return np.array([vx, vy, vz])

    def on_reset(self, env):
        """See base class."""
        self._current_base_velocity = np.array(self._robot.GetBaseVelocity())
        self._last_base_velocity = np.array(self._robot.GetBaseVelocity())
        self._current_yaw = self._robot.GetBaseRollPitchYaw()[2]
        self._last_yaw = self._robot.GetBaseRollPitchYaw()[2]

    def on_step(self, env):
        """See base class."""
        self._last_base_velocity = self._current_base_velocity
        self._current_base_velocity = np.array(self._robot.GetBaseVelocity())
        self._last_yaw = self._current_yaw
        self._current_yaw = self._robot.GetBaseRollPitchYaw()[2]


class BaseDisplacementSensor(sensor.BoxSpaceSensor):
    """A sensor that reads displacement of robot base."""

    def __init__(
        self,
        lower_bound: _FLOAT_OR_ARRAY = -0.1,
        upper_bound: _FLOAT_OR_ARRAY = 0.1,
        convert_to_local_frame: bool = False,
        name: typing.Text = "BaseDisplacement",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs BaseDisplacementSensor.

        Args:
          lower_bound: the lower bound of the base displacement
          upper_bound: the upper bound of the base displacement
          convert_to_local_frame: whether to project dx, dy to local frame based on
            robot's current yaw angle. (Note that it's a projection onto 2D plane,
            and the roll, pitch of the robot is not considered.)
          name: the name of the sensor
          dtype: data type of sensor value
        """

        self._channels = ["x", "y", "z"]
        self._num_channels = len(self._channels)

        super(BaseDisplacementSensor, self).__init__(
            name=name,
            shape=(self._num_channels,),
            lower_bound=np.array([lower_bound] * 3),
            upper_bound=np.array([upper_bound] * 3),
            dtype=dtype,
        )

        datatype = [("{}_{}".format(name, channel), self._dtype) for channel in self._channels]
        self._datatype = datatype
        self._convert_to_local_frame = convert_to_local_frame

        self._last_yaw = 0
        self._last_base_position = np.zeros(3)
        self._current_yaw = 0
        self._current_base_position = np.zeros(3)

    def get_channels(self) -> typing.Iterable[typing.Text]:
        """Returns channels (displacement in x, y, z direction)."""
        return self._channels

    def get_num_channels(self) -> int:
        """Returns number of channels."""
        return self._num_channels

    def get_observation_datatype(self) -> _DATATYPE_LIST:
        """See base class."""
        return self._datatype

    def _get_observation(self) -> _ARRAY:
        """See base class."""
        dx, dy, dz = self._current_base_position - self._last_base_position
        if self._convert_to_local_frame:
            dx_local = np.cos(self._last_yaw) * dx + np.sin(self._last_yaw) * dy
            dy_local = -np.sin(self._last_yaw) * dx + np.cos(self._last_yaw) * dy
            return np.array([dx_local, dy_local, dz])
        else:
            return np.array([dx, dy, dz])

    def on_reset(self, env):
        """See base class."""
        self._current_base_position = np.array(self._robot.GetBasePosition())
        self._last_base_position = np.array(self._robot.GetBasePosition())
        self._current_yaw = self._robot.GetBaseRollPitchYaw()[2]
        self._last_yaw = self._robot.GetBaseRollPitchYaw()[2]

    def on_step(self, env):
        """See base class."""
        self._last_base_position = self._current_base_position
        self._current_base_position = np.array(self._robot.GetBasePosition())
        self._last_yaw = self._current_yaw
        self._current_yaw = self._robot.GetBaseRollPitchYaw()[2]


class IMUSensor(sensor.BoxSpaceSensor):
    """An IMU sensor that reads orientations and angular velocities."""

    def __init__(
        self,
        channels: typing.Iterable[typing.Text] = None,
        noisy_reading: bool = True,
        lower_bound: _FLOAT_OR_ARRAY = None,
        upper_bound: _FLOAT_OR_ARRAY = None,
        name: typing.Text = "IMU",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs IMUSensor.

        It generates separate IMU value channels, e.g. IMU_R, IMU_P, IMU_dR, ...

        Args:
          channels: value channels wants to subscribe. A upper letter represents
            orientation and a lower letter represents angular velocity. (e.g. ['R',
            'P', 'Y', 'dR', 'dP', 'dY'] or ['R', 'P', 'dR', 'dP'])
          noisy_reading: whether values are true observations
          lower_bound: the lower bound IMU values
            (default: [-2pi, -2pi, -2000pi, -2000pi])
          upper_bound: the lower bound IMU values
            (default: [2pi, 2pi, 2000pi, 2000pi])
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._channels = channels if channels else ["R", "P", "dR", "dP"]
        self._num_channels = len(self._channels)
        self._noisy_reading = noisy_reading

        # Compute the default lower and upper bounds
        if lower_bound is None and upper_bound is None:
            lower_bound = []
            upper_bound = []
            for channel in self._channels:
                if channel in ["R", "P", "Y"]:
                    lower_bound.append(-2.0 * np.pi)
                    upper_bound.append(2.0 * np.pi)
                elif channel in ["Rcos", "Rsin", "Pcos", "Psin", "Ycos", "Ysin"]:
                    lower_bound.append(-1.0)
                    upper_bound.append(1.0)
                elif channel in ["dR", "dP", "dY"]:
                    lower_bound.append(-2000.0 * np.pi)
                    upper_bound.append(2000.0 * np.pi)

        super(IMUSensor, self).__init__(
            name=name,
            shape=(self._num_channels,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

        # Compute the observation_datatype
        datatype = [("{}_{}".format(name, channel), self._dtype) for channel in self._channels]

        self._datatype = datatype

    def get_channels(self) -> typing.Iterable[typing.Text]:
        return self._channels

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_observation_datatype(self) -> _DATATYPE_LIST:
        """Returns box-shape data type."""
        return self._datatype

    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            rpy = self._robot.GetBaseRollPitchYaw()
            drpy = self._robot.GetBaseRollPitchYawRate()
        else:
            rpy = self._robot.GetTrueBaseRollPitchYaw()
            drpy = self._robot.GetTrueBaseRollPitchYawRate()

        assert len(rpy) >= 3, rpy
        assert len(drpy) >= 3, drpy

        observations = np.zeros(self._num_channels)
        for i, channel in enumerate(self._channels):
            if channel == "R":
                observations[i] = rpy[0]
            if channel == "Rcos":
                observations[i] = np.cos(rpy[0])
            if channel == "Rsin":
                observations[i] = np.sin(rpy[0])
            if channel == "P":
                observations[i] = rpy[1]
            if channel == "Pcos":
                observations[i] = np.cos(rpy[1])
            if channel == "Psin":
                observations[i] = np.sin(rpy[1])
            if channel == "Y":
                observations[i] = rpy[2]
            if channel == "Ycos":
                observations[i] = np.cos(rpy[2])
            if channel == "Ysin":
                observations[i] = np.sin(rpy[2])
            if channel == "dR":
                observations[i] = drpy[0]
            if channel == "dP":
                observations[i] = drpy[1]
            if channel == "dY":
                observations[i] = drpy[2]
        return observations


class BasePositionSensor(sensor.BoxSpaceSensor):
    """A sensor that reads the base position of the Minitaur robot."""

    def __init__(
        self,
        lower_bound: _FLOAT_OR_ARRAY = -100,
        upper_bound: _FLOAT_OR_ARRAY = 100,
        name: typing.Text = "BasePosition",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs BasePositionSensor.

        Args:
          lower_bound: the lower bound of the base position of the robot.
          upper_bound: the upper bound of the base position of the robot.
          name: the name of the sensor
          dtype: data type of sensor value
        """
        super(BasePositionSensor, self).__init__(
            name=name,
            shape=(3,),  # x, y, z
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def _get_observation(self) -> _ARRAY:
        return self._robot.GetBasePosition()


class PoseSensor(sensor.BoxSpaceSensor):
    """A sensor that reads the (x, y, theta) of a robot."""

    def __init__(
        self,
        lower_bound: _FLOAT_OR_ARRAY = -100,
        upper_bound: _FLOAT_OR_ARRAY = 100,
        name: typing.Text = "PoseSensor",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs PoseSensor.

        Args:
          lower_bound: the lower bound of the pose of the robot.
          upper_bound: the upper bound of the pose of the robot.
          name: the name of the sensor.
          dtype: data type of sensor value.
        """
        super(PoseSensor, self).__init__(
            name=name,
            shape=(3,),  # x, y, orientation
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def _get_observation(self) -> _ARRAY:
        return np.concatenate(
            (
                self._robot.GetBasePosition()[:2],
                (self._robot.GetTrueBaseRollPitchYaw()[2],),
            )
        )


class FootForceSensor(sensor.BoxSpaceSensor):
    """A sensor that reads foot force from the robot."""

    def __init__(
        self,
        num_legs: int,
        noisy_reading: bool = True,
        lower_bound: _FLOAT_OR_ARRAY = 0,
        upper_bound: _FLOAT_OR_ARRAY = 200,
        name: typing.Text = "FootForce",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs FootForceSensor.

        Args:
          num_legs: the number of legs in the robot
          noisy_reading: whether values are true observations
          lower_bound: the lower bound of the normal force
          upper_bound: the upper bound of the normal force
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_legs = num_legs
        self._noisy_reading = noisy_reading
        super(FootForceSensor, self).__init__(
            name=name,
            shape=(self._num_legs,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def _get_observation(self) -> _ARRAY:
        return self._robot.GetFootForces()


class FootContactSensor(sensor.BoxSpaceSensor):
    """A sensor that reads foot contact from the robot."""

    def __init__(
        self,
        num_legs: int,
        lower_bound: _FLOAT_OR_ARRAY = 0,
        upper_bound: _FLOAT_OR_ARRAY = 1,
        name: typing.Text = "FootContact",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs FootContactSensor.

        Args:
          num_legs: the number of legs in the robot
          noisy_reading: whether values are true observations
          lower_bound: the lower bound of the normal contact
          upper_bound: the upper bound of the normal contact
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._num_legs = num_legs
        super(FootContactSensor, self).__init__(
            name=name,
            shape=(self._num_legs,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def _get_observation(self) -> _ARRAY:
        return self._robot.GetFootContacts()


class LocalTerrainDepthByAngleSensor(sensor.BoxSpaceSensor):
    """A sensor that gets the depth from the robot to the ground"""

    def __init__(
        self,
        noisy_reading: bool = True,
        grid_angle: typing.Tuple[float] = (0.1, 0.1),
        grid_size: typing.Tuple[int] = (10, 10),
        transform_angle: typing.Tuple[float] = (0, 0),
        lower_bound: _FLOAT_OR_ARRAY = 0.0,
        upper_bound: _FLOAT_OR_ARRAY = 8.0,
        name: typing.Text = "LocalTerrainDepthByAngle",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
        use_filter: bool = False,
        filter_every: int = 1,
    ) -> None:
        """Constructs LocalTerrainDepthByAngleSensor.

        Args:
          grid_unit: Side length of one square in the grid
          grid_size: Number of squares along one side of grid
          lower_bound: the lower bound of the terrain view.
          upper_bound: the upper bound of the terrain view.
          name: the name of the sensor.
          dtype: data type of sensor value.
          use_filter: Whether to apply filtering,
          filter_every: Filter will receive observation every
            (filter_every) simulation steps. Only used if
            use_filter is True.
        """
        self._env = None
        self._noisy_reading = noisy_reading
        self.grid_angle = grid_angle
        self.grid_size = grid_size
        self.transform_angle = transform_angle
        self.use_filter = use_filter
        self.filter_every = filter_every

        shape = (1, grid_size[0], grid_size[1])
        super(LocalTerrainDepthByAngleSensor, self).__init__(
            name=name,
            shape=shape,
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def _get_observation(self) -> _ARRAY:
        """Returns (and maybe filters) the local distances to ground"""
        if self.use_filter:
            return self._filtered_heightmap
        else:
            return self._get_heightmap()

    def _get_heightmap(self):
        """Returns the local distances to ground"""
        heightmap = self._robot.GetLocalTerrainDepthByAngle(
            grid_angle=self.grid_angle, grid_size=self.grid_size, transform_angle=self.transform_angle
        ).reshape(1, self.grid_size[0], self.grid_size[1])
        # Add noise
        if self._noisy_reading:
            heightmap = heightmap + np.random.normal(scale=0.01, size=heightmap.shape)
        # Clip readings
        heightmap = np.minimum(np.maximum(heightmap, 0.1), 8.0)
        return heightmap

    def on_simulation_step(self):
        if self.use_filter:
            if not hasattr(self, "filter"):
                # First time setup of filter
                self.filter = action_filter.ActionFilterButter(
                    lowcut=[0.0],
                    highcut=[6.0],  # Hz
                    sampling_rate=1 / (self._robot.time_step * self.filter_every),
                    order=1,
                    num_joints=np.prod(self.get_shape()),
                )
                self.filter_counter = 0

            if self.filter_counter % self.filter_every == 0:
                heightmap = self._get_heightmap()
                orig_shape = heightmap.shape
                filtered_flat_heightmap = self.filter.filter(heightmap.reshape(-1))
                self._filtered_heightmap = filtered_flat_heightmap.reshape(orig_shape)

            self.filter_counter += 1
