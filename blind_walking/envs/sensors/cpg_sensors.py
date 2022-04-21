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
    "walk": lambda phase: 2 * (phase > 0).astype(float) - 1,
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
        gait_frequency_lower: float = DEFAULT_GAIT_FREQUENCY,
        gait_frequency_upper: float = DEFAULT_GAIT_FREQUENCY,
        duty_factor_lower: float = DEFAULT_DUTY_FACTOR,
        duty_factor_upper: float = DEFAULT_DUTY_FACTOR,
        obs_steps_ahead: typing.List[int] = [0],
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
            alpha=10.0,
            beta=DEFAULT_DUTY_FACTOR,
            gamma=50.0,
            period=1 / DEFAULT_GAIT_FREQUENCY,
            dt=0.030,  # 0.03 seconds = 0.001 sim_time_step * 30 action_repeat
        )

        self._gait_name = gait_name
        self._phase_offset = phase_offsets[gait_name]
        self._get_foot_contact = foot_contact_fn[gait_name]
        self._gait_frequency_range = (gait_frequency_lower, gait_frequency_upper)
        self._duty_factor_range = (duty_factor_lower, duty_factor_upper)
        self._obs_steps_ahead = obs_steps_ahead

        self.cpg_system = CPGSystem(
            params=params,
            coupling_strength=1,
            desired_phase_offsets=self._phase_offset,
            initial_state=CPGSystem.sample_initial_state(self._phase_offset),
        )

        super(ReferenceGaitSensor, self).__init__(
            name=name,
            shape=(4 * len(self._obs_steps_ahead),),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )
        self._reset()
        print(f"Init CPG gait={self.get_gait_name()}, duty_factor={self.get_duty_factor()}, period={self.get_period()}")

    def on_step(self, env):
        del env

        self.cpg_system.step()
        self._current_phase = self.cpg_system.get_phase()
        obs = self._get_foot_contact(self._current_phase)
        self._obs_history_buffer.append(obs)
        self._obs_history_buffer.pop(0)

    def _reset(self):
        # Clear the history buffer
        self._obs_history_buffer = []
        # Reset CPG to a random state
        gait_frequency = np.random.uniform(self._gait_frequency_range[0], self._gait_frequency_range[1])
        self.set_period(1 / gait_frequency)
        duty_factor = np.random.uniform(self._duty_factor_range[0], self._duty_factor_range[1])
        self.set_duty_factor(duty_factor)
        self.cpg_system.set_state(CPGSystem.sample_initial_state(self._phase_offset))
        self._current_phase = self.cpg_system.get_phase()

        # Initialize the history buffer
        for i in range(10):
            self.cpg_system.step()
            self._current_phase = self.cpg_system.get_phase()
            obs = self._get_foot_contact(self._current_phase)
            self._obs_history_buffer.append(obs)

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
        obses = [self._obs_history_buffer[obs_idx] for obs_idx in self._obs_steps_ahead]
        return np.concatenate(obses)

    def get_current_reference_state(self):
        """
        Gets the correct reference foot contact state at the current time.
        """
        return self._obs_history_buffer[0]

    def get_period(self):
        return self.cpg_system.params.period

    def set_period(self, value):
        self.cpg_system.params.period = value

    def get_duty_factor(self):
        return self.cpg_system.params.beta

    def set_duty_factor(self, value):
        self.cpg_system.params.beta = value

    def get_gait_name(self):
        return self._gait_name

    def set_gait_name(self, value):
        self._gait_name = value
        self._get_foot_contact = foot_contact_fn[value]
        # Note: this currently does not account for smooth transitions


class ReferenceFootPositionSensor(sensor.BoxSpaceSensor):
    pass
