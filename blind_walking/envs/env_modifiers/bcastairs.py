import pybullet as p
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier

# Dimensions of BCA stairs
# Fabrication, Long Wooden Staircase - 7steps
# Size: 610H * 1200D * 2440L mm
# Color: Black emulsion coated paint
# Step Height/rise: 17.5 cm
# Step Depth/run: 27.5 cm
# Step Width: 1.5 m

# Carpet platform
# Size: 1.5mL x 1.5mD x 0.61m Height

# Its a up, platform then down structure. Can change to just up and platform. Not crucial.

# top lvl/ platform
boxHalfLength = 0.75
boxHalfWidth = 0.75
boxHalfHeight = 0.305


class BCAStairs(EnvModifier):
    def __init__(self):
        self.stepShape = 0
        self.steps = []
        self.base_pos = [0, 0, 0]
        super().__init__()

    def _generate(self, env, start_x=3, num_steps=7, step_rise=0.175, step_run=0.275, friction=0.5):
        env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_RENDERING, 0)

        stepShape = env.pybullet_client.createCollisionShape(
            p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight]
        )
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

        # Create upwards stairs
        base_pos = [start_x, 0, step_rise - boxHalfHeight]
        self.base_pos = base_pos
        for i in range(num_steps):
            step = env.pybullet_client.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=stepShape, basePosition=base_pos, baseOrientation=[0, 0, 0, 1]
            )
            self.steps.append(step)
            env.pybullet_client.changeDynamics(step, -1, lateralFriction=friction)
            env.pybullet_client.changeVisualShape(step, -1, rgbaColor=colors[i % len(colors)])
            base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, step_rise])]

        # Create downwards stairs
        base_pos = [sum(x) for x in zip(base_pos, [boxHalfLength * 2 - step_run, 0, -step_rise])]
        for i in range(num_steps):
            step = env.pybullet_client.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=stepShape, basePosition=base_pos, baseOrientation=[0, 0, 0, 1]
            )
            self.steps.append(step)
            env.pybullet_client.changeDynamics(step, -1, lateralFriction=friction)
            env.pybullet_client.changeVisualShape(step, -1, rgbaColor=colors[(-i - 1) % len(colors)])
            base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, -step_rise])]

        self.stepShape = stepShape
        env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_RENDERING, 1)

    def _reset(self, env, step_rise=0.175, step_run=0.275):
        # Do not change the box shape but change the position of the steps
        base_pos = self.base_pos
        for i, step in enumerate(self.steps):
            p.resetBasePositionAndOrientation(step, base_pos, [0, 0, 0, 1])
            if i < len(self.steps) / 2 - 1:
                base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, step_rise])]
            elif i == len(self.steps) / 2 - 1:
                base_pos = [sum(x) for x in zip(base_pos, [boxHalfLength * 2, 0, 0])]
            else:
                base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, -step_rise])]
