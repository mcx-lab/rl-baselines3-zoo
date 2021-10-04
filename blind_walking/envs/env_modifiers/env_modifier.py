from abc import ABC, abstractmethod


class EnvModifier(ABC):
    def __init__(self, adjust_position=[0, 0, 0], deformable=False):
        self.adjust_position = adjust_position  # Adjust initial position of robot
        self.deformable = deformable  # Whether the modifier requires deformable physics

    @abstractmethod
    def _generate(self, env, **kwargs):
        pass
