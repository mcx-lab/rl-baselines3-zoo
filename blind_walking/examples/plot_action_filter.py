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
from scipy import signal

import utils.import_envs  # noqa: F401 pytype: disable=import-error


def _all_columns_equal(x: np.ndarray):
    return np.all(x == x[0, :])


def plot_frequency_response(filter: ActionFilterButter, savepath: str):
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
    plt.savefig(savepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", default=False, help="Record video")
    parser.add_argument("--savedir", type=str, help="Directory to save action filter plot", default="plots")
    parser.add_argument(
        "--filename", type=str, help="Filename to save plot, e.g. 'plot.png'", default="action_filter_frequency_response.png"
    )
    args = parser.parse_args()

    grid_size = (20, 1)
    grid_unit = 0.05
    grid_transform = (0.15, 0)

    # Environment parameters
    robot_sensor_list = []
    env_sensor_list = [
        # We don't need any sensors for plotting the filter
        # However, the gym env itself requires at least one sensor
        environment_sensors.LocalTerrainDepthSensor(grid_size=grid_size, grid_unit=grid_unit, transform=grid_transform),
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
    savepath = str(savedir / args.filename)
    plot_frequency_response(env.robot._action_filter, savepath)


if __name__ == "__main__":
    main()
