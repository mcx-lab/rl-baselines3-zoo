from blind_walking.envs.env_modifiers.env_modifier import EnvModifier
from blind_walking.envs.env_modifiers.stairs import Stairs
from blind_walking.envs.env_modifiers.heightfield import HeightField


class TrainCourse(EnvModifier):
    def __init__(self):
        super().__init__()
        self.stairs = Stairs()
        self.heightfield = HeightField()

    def _generate(self, env):
        # Generate stairs after some flatground
        self.stairs._generate(env, start_x=7)
        # Generate heightfield after stairs \
        # offset 9 for heightfield length, 7 for flatground, \
        # 4 for stairs length, 2 for additional flatground
        self.heightfield._generate(env, start_x=22)
