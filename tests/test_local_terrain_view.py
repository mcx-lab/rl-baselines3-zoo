import numpy as np
import gym

from blind_walking.envs import locomotion_gym_env
from blind_walking.envs import locomotion_gym_config
from blind_walking.envs.env_wrappers import observation_dictionary_split_by_encoder_wrapper as obs_split_wrapper
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.env_wrappers import trajectory_generator_wrapper_env
from blind_walking.envs.env_wrappers import simple_openloop
from blind_walking.envs.env_modifiers import heightfield, stairs
from blind_walking.envs.tasks import forward_task, forward_task_pos
from blind_walking.envs.sensors import robot_sensors, environment_sensors
from blind_walking.robots import a1
from blind_walking.robots import laikago
from blind_walking.robots import robot_config
from unittest import TestCase


# Create a copy of _build_regular_env since this changes frequently in our codebase
def _build_regular_env(robot_class,
                      motor_control_mode,
                      enable_rendering=False,
                      on_rack=False,
                      action_limit=(0.75, 0.75, 0.75),
                      wrap_trajectory_generator=True):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  sim_params.enable_action_filter = True
  sim_params.enable_clip_motor_commands = True
  sim_params.robot_on_rack = on_rack

  gym_config = locomotion_gym_config.LocomotionGymConfig(
    simulation_parameters=sim_params)

  robot_sensor_list = [
    robot_sensors.BaseVelocitySensor(convert_to_local_frame=True, exclude_z=True),
    robot_sensors.IMUSensor(channels=['R', 'P', 'Y', 'dR', 'dP', 'dY']),
    robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
    robot_sensors.MotorVelocitySensor(num_motors=a1.NUM_MOTORS),
  ]

  env_sensor_list = [
    environment_sensors.LastActionSensor(num_actions=a1.NUM_MOTORS),
    environment_sensors.TargetPositionSensor(),
  ]

  env_randomizer_list = []

  env_modifier_list = []

  task = forward_task_pos.ForwardTask()

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                            robot_class=robot_class,
                                            robot_sensors=robot_sensor_list,
                                            env_sensors=env_sensor_list,
                                            task=task,
                                            env_randomizers=env_randomizer_list,
                                            env_modifiers=env_modifier_list,)

  env = obs_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  if (motor_control_mode
      == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
    if robot_class == laikago.Laikago:
      env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
          env,
          trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
              action_limit=action_limit))
    elif robot_class == a1.A1:
      env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
          env,
          trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
              action_limit=action_limit))
  return env

class DummyA1GymEnv(gym.Env):
  """ An frozen version of A1GymEnv that can be used for testing """
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self,
               action_limit=(0.3, 0.3, 0.3),
               render=False,
               on_rack=False):
    self._env = _build_regular_env(
        a1.A1,
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        enable_rendering=render,
        action_limit=action_limit,
        on_rack=on_rack)
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space

  def step(self, action):
    return self._env.step(action)

  def reset(self):
    return self._env.reset()

  def close(self):
    self._env.close()

  def render(self, mode):
    return self._env.render(mode)

  def __getattr__(self, attr):
    return getattr(self._env, attr)

def test_local_terrain_view_sensor():
    env = DummyA1GymEnv()
    grid_size = 32
    local_terrain_view = env.robot.GetLocalTerrainViewBatch(grid_unit = 0.1, grid_size = grid_size)
    assert local_terrain_view.shape == (grid_size, grid_size)

    import matplotlib.pyplot as plt
    cs = [i for i in range(grid_size)]
    coords = [(x,y) for x in cs for y in cs]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    plt.figure()
    plt.scatter(xs, ys, c = local_terrain_view)
    plt.savefig('plot.png')