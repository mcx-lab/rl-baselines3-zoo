""" Sensors related to CPG """
import csv
import typing

import numpy as np
import torch
from blind_walking.envs.sensors import sensor
from blind_walking.envs.sensors.environment_sensors import _ARRAY, _FLOAT_OR_ARRAY
from blind_walking.envs.utilities.cpg import CPG, CPGParameters, CPGSystem


class ReferenceGaitSensor(sensor.BoxSpaceSensor):
    """A sensor that reports whether each foot should be in contact with the ground.

    Reference foot contact states are decoded according to phases.
    Phases are generated internally using a CPG.
    """

    def __init__(
        self,
        lower_bound: _FLOAT_OR_ARRAY = -np.pi,
        upper_bound: _FLOAT_OR_ARRAY = np.pi,
        name: typing.Text = "ReferenceGait",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs ReferenceGaitSensor.
        Args:
          lower_bound: the lower bound of the phase
          upper_bound: the upper bound of the phase
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        params = CPGParameters(
            a=1.0,
            b=50.0,
            mu=1.0,
            alpha=50.0,
            beta=0.75,
            gamma=50.0,
            period=1.0,  # 1 second to complete a cycle
            dt=0.030,  # 0.03 seconds = 0.001 sim_time_step * 30 action_repeat
        )

        self._walk_offsets = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        self._walk_get_foot_contact = lambda phase: 2 * np.logical_and(phase < np.pi / 2, phase > 0).astype(float) - 1

        self._get_foot_contact = self._walk_get_foot_contact
        self.cpg_system = CPGSystem(
            params=params,
            coupling_strength=1,
            desired_phase_offsets=self._walk_offsets,
            initial_state=CPGSystem.sample_initial_state(self._walk_offsets),
        )

        super(ReferenceGaitSensor, self).__init__(
            name=name,
            shape=(4,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_step(self, env):
        del env

        self.cpg_system.step()
        self._current_phase = self.cpg_system.get_phase()

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

        # Reset to a random state
        self.cpg_system.set_state(CPGSystem.sample_initial_state(self._walk_offsets))
        self._current_phase = self.cpg_system.get_phase()

    def _get_observation(self) -> _ARRAY:
        """Returns np.ndarray of shape (4,)

        obs[i] = 1 iff foot[i] should be in contact with ground and -1 otherwise"""
        return self._get_foot_contact(self._current_phase)


class ReferenceFootPositionSensor(sensor.BoxSpaceSensor):
    pass
