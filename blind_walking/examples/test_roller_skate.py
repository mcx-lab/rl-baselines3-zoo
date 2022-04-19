"""Example of whole body controller on A1 robot."""
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pybullet  # pytype:disable=import-error
import pybullet_data
from absl import app, flags, logging
from blind_walking.data import utils
from blind_walking.robots import a1_wheeled, robot_config
from pybullet_utils import bullet_client

# flake8: noqa

flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 30.0, "maximum time to run the robot.")
FLAGS = flags.FLAGS

_NUM_SIMULATION_ITERATION_STEPS = 300
_MAX_TIME_SECONDS = 30.0


def main(argv):
    """Runs the locomotion controller example."""
    del argv  # unused

    # Construct simulator
    if FLAGS.show_gui:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(utils.getDataPath())
    print(utils.getDataPath())
    p.loadURDF("terrain/plane.urdf")

    # Construct robot class:
    robot = a1_wheeled.A1(
        p,
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        enable_action_interpolation=False,
        reset_time=2,
        time_step=0.002,
        action_repeat=1,
    )

    if FLAGS.logdir:
        logdir = os.path.join(FLAGS.logdir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(logdir)

    start_time = robot.GetTimeSinceReset()
    current_time = start_time
    fixed_action = np.array([0, 0.9, -1.8] * 4)

    # Apply forward force to robot
    robot._pybullet_client.applyExternalForce(
        robot.quadruped, -1, [10000, 0, 0], [0, 0, 0], flags=robot._pybullet_client.LINK_FRAME
    )

    states, actions = [], []
    while current_time - start_time < FLAGS.max_time_secs:
        start_time_robot = current_time
        start_time_wall = time.time()

        robot.Step(fixed_action)
        current_time = robot.GetTimeSinceReset()
        expected_duration = current_time - start_time_robot
        actual_duration = time.time() - start_time_wall
        if actual_duration < expected_duration:
            time.sleep(expected_duration - actual_duration)

    if FLAGS.logdir:
        np.savez(os.path.join(logdir, "action.npz"), action=actions)
        pickle.dump(states, open(os.path.join(logdir, "states.pkl"), "wb"))
        logging.info("logged to: {}".format(logdir))


if __name__ == "__main__":
    app.run(main)
