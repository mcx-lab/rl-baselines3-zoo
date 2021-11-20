from blind_walking.envs.gym_envs import A1GymEnv
from blind_walking.envs.sensors import environment_sensors, robot_sensors


def count_function_calls_wrapper(f):
    def wrapped_f(*args, **kwargs):
        wrapped_f.calls += 1
        return f(*args, **kwargs)

    wrapped_f.calls = 0
    return wrapped_f


def test_local_terrain_depth_by_angle_sensor_filtered():
    """Test that the on_simulation_step is called 30 times per action"""
    sensor = robot_sensors.LocalTerrainDepthByAngleSensor(
        grid_size=(10, 1),
        grid_angle=(0.1, 0),
        transform_angle=(-0.8, 0),
        noisy_reading=True,
        name="depthmiddle",
        use_filter=True,
        filter_every=7,
    )

    env = A1GymEnv(
        robot_sensor_list=[sensor],
        env_sensor_list=[
            environment_sensors.ForwardTargetPositionSensor(max_distance=0.02),
        ],
    )
    env.reset()
    sensor.on_simulation_step = count_function_calls_wrapper(sensor.on_simulation_step)
    env.robot.GetLocalTerrainDepthByAngle = count_function_calls_wrapper(env.robot.GetLocalTerrainDepthByAngle)
    env.step(env.action_space.sample())
    assert sensor.filter_every == 7
    assert sensor.on_simulation_step.calls == env._num_action_repeat
    assert env.robot.GetLocalTerrainDepthByAngle.calls == env._num_action_repeat // sensor.filter_every


if __name__ == "__main__":
    test_local_terrain_depth_by_angle_sensor_filtered()
