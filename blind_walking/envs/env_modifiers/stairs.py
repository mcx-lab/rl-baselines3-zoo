import pybullet as p
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier


class Stairs(EnvModifier):
    def __init__(self):
        super().__init__()

    def _generate(self, env, start_x=3, num_steps=5, step_rise=0.1, step_run=0.2, friction=0.5):
        env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_RENDERING, 0)

        boxHalfLength = 0.5
        boxHalfWidth = 2.5
        boxHalfHeight = step_rise * 0.5
        stepShape = env.pybullet_client.createCollisionShape(
            p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight]
        )
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

        # Create upwards stairs
        base_pos = [start_x, 0, boxHalfHeight]
        for i in range(num_steps):
            step = env.pybullet_client.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=stepShape,
                basePosition=base_pos,
                baseOrientation=[0, 0, 0, 1],
            )
            env.pybullet_client.changeDynamics(step, -1, lateralFriction=friction)
            env.pybullet_client.changeVisualShape(step, -1, rgbaColor=colors[i % len(colors)])
            base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, step_rise])]

        # Create downwards stairs
        base_pos = [sum(x) for x in zip(base_pos, [boxHalfLength * 2 - step_run, 0, -step_rise])]
        for i in range(num_steps):
            step = env.pybullet_client.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=stepShape,
                basePosition=base_pos,
                baseOrientation=[0, 0, 0, 1],
            )
            env.pybullet_client.changeDynamics(step, -1, lateralFriction=friction)
            env.pybullet_client.changeVisualShape(step, -1, rgbaColor=colors[(-i - 1) % len(colors)])
            base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, -step_rise])]

        env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_RENDERING, 1)
