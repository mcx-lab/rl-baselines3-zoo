""" Sensors related to CPG """
import csv
import typing

import numpy as np
import torch
from blind_walking.envs.sensors import sensor
from blind_walking.envs.sensors.environment_sensors import _ARRAY, _FLOAT_OR_ARRAY
from blind_walking.envs.utilities.cpg import CPG, CPGParameters, CPGSystem

# Foot order: ['FR', 'FL', 'RR', 'RL']

phase_offsets = {
    "walk": np.array([3 * np.pi / 2, np.pi / 2, np.pi, 0]),
    "trot": np.array([np.pi, 0, 0, np.pi]),
    "pace": np.array([np.pi, 0, np.pi, 0]),
    "bound": np.array([np.pi, np.pi, 0, 0]),
}

foot_contact_fn = {
    "walk": lambda phase: 2 * np.logical_or(phase > np.pi / 2, phase < 0).astype(float) - 1,
    "trot": lambda phase: 2 * (phase > 0).astype(int) - 1,
    "pace": lambda phase: 2 * (phase > 0).astype(int) - 1,
    "bound": lambda phase: 2 * (phase > 0).astype(int) - 1,
}

DEFAULT_GAIT_FREQUENCY = 1.5  # Hz
DEFAULT_DUTY_FACTOR = 0.5


class ReferenceGaitSensor(sensor.BoxSpaceSensor):
    """A sensor that reports whether each foot should be in contact with the ground.

    Reference foot contact states are decoded according to phases.
    Phases are generated internally using a CPG.
    """

    def __init__(
        self,
        gait_name: str,
        gait_frequency: float = None,  # Hz
        duty_factor: float = None,
        deterministic: bool = False,
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
        if gait_frequency is None:
            gait_frequency = DEFAULT_GAIT_FREQUENCY
        if duty_factor is None:
            duty_factor = DEFAULT_DUTY_FACTOR

        params = CPGParameters(
            a=1.0,
            b=50.0,
            mu=1.0,
            alpha=10.0,
            beta=duty_factor,
            gamma=50.0,
            period=1 / gait_frequency,
            dt=0.030,  # 0.03 seconds = 0.001 sim_time_step * 30 action_repeat
        )

        self._gait_name = gait_name
        self._phase_offset = phase_offsets[gait_name]
        self._get_foot_contact = foot_contact_fn[gait_name]
        self._deterministic = deterministic

        self.cpg_system = CPGSystem(
            params=params,
            coupling_strength=1,
            desired_phase_offsets=self._phase_offset,
            initial_state=CPGSystem.sample_initial_state(self._phase_offset),
        )

        super(ReferenceGaitSensor, self).__init__(
            name=name,
            shape=(4,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )
        self._reset()

    def on_step(self, env):
        del env

        self.cpg_system.step()
        self._current_phase = self.cpg_system.get_phase()

    def _reset(self):
        # Reset to a random state
        if not self._deterministic:
            self.set_period(np.random.uniform(low=1.0, high=2.0))
        self.cpg_system.set_state(CPGSystem.sample_initial_state(self._phase_offset))
        self._current_phase = self.cpg_system.get_phase()

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env
        self._reset()

    def _get_observation(self) -> _ARRAY:
        """Returns np.ndarray of shape (4,)

        obs[i] = 1 iff foot[i] should be in contact with ground and -1 otherwise"""
        return self._get_foot_contact(self._current_phase)

    def get_period(self):
        return self.cpg_system.params.period

    def set_period(self, value):
        self.cpg_system.params.period = value


class ReferenceFootPositionSensor(sensor.BoxSpaceSensor):
    pass
