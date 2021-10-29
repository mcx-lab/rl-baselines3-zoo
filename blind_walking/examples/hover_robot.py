"""Script for building a hovering robot and collecting heightmap data."""

import argparse
import os
import gym
import numpy as np
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from gym.wrappers import Monitor
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.sensors import environment_sensors
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier
from blind_walking.envs.env_modifiers.stairs import Stairs, boxHalfLength, boxHalfWidth
from blind_walking.envs.env_modifiers.heightfield import HeightField
from enjoy import Logger


class MultipleTerrain(EnvModifier):
    def __init__(self):
        super().__init__()
        self.hf = HeightField()
        self.stairs = Stairs()

        self.start_x = 5
        # Stairs parameters
        self.step_rise = 0.05
        self.num_steps = 5
        self.stair_gap = 1.5
        self.step_run = 0.3
        self.stair_length = (self.num_steps - 1) * self.step_run * 2 + boxHalfLength * 2 * 2
        # Heightfield parameters
        self.hf_length = 18

    def _generate(self, env):
        start_x = self.start_x
        self.stairs._generate(env, start_x=start_x, num_steps=self.num_steps, step_rise=self.step_rise, step_run=self.step_run)
        start_x += self.stair_length + self.hf_length / 2
        self.hf._generate(env, start_x=start_x, heightPerturbationRange=0.04)

    def get_z_position(self, x, y):
        """Get z position for hovering robot at the xy-coord."""
        """NOTE: Position checking are currently hardcoded"""
        if x > self.start_x and x < self.start_x + self.stair_length:
            # On the stairs
            x_stairs = x - self.start_x
            if x_stairs < self.step_run * (self.num_steps - 1):
                # On the ascending stairs
                step_on = 1 + x_stairs // self.step_run
            elif x_stairs < self.step_run * (self.num_steps - 1) + boxHalfLength * 2 * 2:
                # On the top of the stairs
                step_on = self.num_steps
            else:
                # On the descending stairs
                x_stairs -= self.step_run * (self.num_steps - 1) + boxHalfLength * 2 * 2
                step_on = self.num_steps - 1 - x_stairs // self.step_run
            z_pos = step_on * self.step_rise
        elif x > self.start_x + self.stair_length and x < self.start_x + self.stair_length + self.hf_length:
            # On uneven terrain
            z_pos = 0.04
        else:
            z_pos = 0
        return z_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", default=False, help="Record video")
    args = parser.parse_args()

    dx = 0.05
    dy = 0

    # Environment parameters
    robot_sensor_list = []
    env_sensor_list = [
        environment_sensors.LocalTerrainViewSensor(grid_size=(20, 1), grid_unit=0.05, transform=(0.15, 0)),
    ]
    env_randomizer_list = []
    env_modifier_list = [MultipleTerrain()]
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
    if args.record:
        video_folder = os.path.dirname(__file__)
        env = Monitor(env, video_folder, force=True)
    env.reset()

    # Create logger
    hm_logger = Logger("heightmap")

    # Move robot across terrain and collect heightmap data
    default_orientation = env.robot._GetDefaultInitOrientation()
    default_position = env.robot._GetDefaultInitPosition()
    zero_action = np.zeros(12)
    position = default_position.copy()
    for _ in range(1000):
        # Update position
        position[0] += dx
        position[1] += dy
        z_pos = env_modifier_list[0].get_z_position(position[0], position[1])
        position[2] = default_position[2] + z_pos
        env.pybullet_client.resetBasePositionAndOrientation(env.robot.quadruped, position, default_orientation)
        obs, _, _, _ = env.step(zero_action)
        # Record heightmap data
        hm_logger.update([obs])

    # Close environment and save data
    env.close()
    hm_logger.save(os.path.dirname(__file__))
