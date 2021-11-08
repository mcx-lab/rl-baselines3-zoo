"""Script for plotting action filter."""

import argparse
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
from blind_walking.envs.env_modifiers.train_course import TrainMultiple
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.sensors import environment_sensors
from blind_walking.robots.action_filter import ActionFilterButter
from numpy.random import MT19937, RandomState, SeedSequence
from scipy import signal

import utils.import_envs  # noqa: F401 pytype: disable=import-error


def _all_columns_equal(x: np.ndarray):
    return np.all(x == x[0, :])


def plot_frequency_response(filter: ActionFilterButter):
    """Visualize the frequency response of an ActionFilterButter"""
    assert _all_columns_equal(filter.b)
    assert _all_columns_equal(filter.a)
    w, h = signal.freqs(filter.b[0], filter.a[0])
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title("Butterworth filter frequency response")
    plt.xlabel("Frequency [radians / second]")
    plt.ylabel("Amplitude [dB]")
    plt.margins(0, 0.1)
    plt.grid(which="both", axis="both")
    plt.axvline(filter.highcut[0] / (2 * np.pi), color="green")  # cutoff frequency


def plot_signal(signal: np.ndarray, label: str):
    time = np.arange(len(signal))
    plt.plot(time, signal, label=label)
    plt.xlabel("Time step")
    plt.ylabel("Signal value")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", default=False, help="Record video")
    parser.add_argument("--savedir", type=str, help="Directory to save action filter plot", default="plots")
    args = parser.parse_args()

    grid_size = (20, 1)
    grid_unit = 0.05
    grid_transform = (0.15, 0)

    # Environment parameters
    robot_sensor_list = []
    env_sensor_list = [
        # We don't need any sensors for plotting the filter
        # However, forward_task_pos.ForwardTask requires this sensor
        environment_sensors.ForwardTargetPositionSensor(max_distance=0.03),
    ]
    env_randomizer_list = []
    env_modifier_list = [TrainMultiple()]
    obs_wrapper = obs_array_wrapper.ObservationDictionaryToArrayWrapper

    # Create environment
    env = gym.make(
        "A1GymEnv-v0",
        robot_sensor_list=robot_sensor_list,
        env_sensor_list=env_sensor_list,
        env_randomizer_list=env_randomizer_list,
        env_modifier_list=env_modifier_list,
        obs_wrapper=obs_wrapper,
    )
    env.reset()

    # Plot action filter frequency ressponse
    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    plot_frequency_response(env.robot._action_filter)
    plt.savefig(str(savedir / "frequency_response.png"))

    # Plot signal output on a test signal
    random_gen = RandomState(MT19937(SeedSequence(0)))
    noise = random_gen.normal(loc=0.0, scale=1.0, size=(12, 1000))
    filtered_noise = np.zeros_like(noise)
    for timestep in range(noise.shape[1]):
        filtered_noise[:, timestep] = env.robot._action_filter.filter(noise[:, timestep])

    # Plot result of action filter on Gaussian noise
    plt.figure()
    plot_signal(noise[0], label="White noise")
    plot_signal(filtered_noise[0], label="Filtered noise")
    plt.legend()
    plt.savefig(str(savedir / "noise_and_filtered.png"))


if __name__ == "__main__":
    main()
