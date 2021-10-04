import pybullet as p
import pybullet_data as pd
import os
import numpy as np
import pyquaternion
import time

import random

random.seed(10)

from blind_walking.envs.env_modifiers.env_modifier import EnvModifier


""" Collapsible Tile Env
Using _generate_field():
Case 1: Front right feet damping platform (Height above ground 0.50m)
Case 2: Front left feed damping platform (Height above ground 0.50m)
Usage: 
self.cp._generate_field(self,
                        case=1,
                        sElasticStiffness=8,
                        sDampingStiffness=1,
                        texture='water_texture' # or 'grass_texture'
                    )
"""


class CollapsibleTile(EnvModifier):
    def __init__(self):
        self.cp_id = 0
        self.textureId = None
        self.color = "G"

        # Store simulation environment tiles id and default values
        self.platform = []
        self.damping_platform = []
        self.collapsible_platform = []
        self.damping_tile_pos = []
        self.collapsible_tile_pos = []
        self.sElasticStiffness = 8
        self.sDampingStiffness = 1
        # Generate map matrix information structure (for generate_soft_env)
        self.map_mat = {
            "obj_id": [],
            "color": [],
            "basePosition": [],
            "mass": [],
            "sElasticStiff": [],
            "sDampingStiff": [],
            "collapsibility": [],
        }
        super().__init__(adjust_position=[0, 0, 0.5], deformable=True)

    def _reset(self, env):
        for index, blockId in enumerate(self.damping_platform):
            env.pybullet_client.resetBasePositionAndOrientation(
                blockId, self.damping_tile_pos[index], [0, 0, 0, 1]
            )
            env.pybullet_client.resetBaseVelocity(blockId, [0, 0, 0], [0, 0, 0])
            if self.textureId:
                env.pybullet_client.changeVisualShape(
                    blockId, -1, textureUniqueId=self.textureId
                )
                env.pybullet_client.changeDynamics(blockId, -1, mass=0.07)
            elif self.color == "O":
                env.pybullet_client.changeVisualShape(
                    blockId, -1, rgbaColor=[1, 0.5, 0, 1], flags=0
                )
                env.pybullet_client.changeDynamics(blockId, -1, mass=0.25)
            elif self.color == "R":
                env.pybullet_client.changeVisualShape(
                    blockId, -1, rgbaColor=[1, 0.25, 0, 1], flags=0
                )
                env.pybullet_client.changeDynamics(blockId, -1, mass=0.07)
        for index, blockId in enumerate(self.collapsible_platform):
            print(f"{index}, {blockId}")
            env.pybullet_client.removeBody(blockId)
            del self.collapsible_platform[index]
            regenerated_body = env.pybullet_client.loadSoftBody(
                "cube.obj",
                basePosition=self.collapsible_tile_pos[index],
                scale=0.25,
                mass=1.0,
                useNeoHookean=0,
                useBendingSprings=1,
                useMassSpring=1,
                springElasticStiffness=self.sElasticStiffness,
                springDampingStiffness=self.sDampingStiffness,
                springDampingAllDirections=1,
                collisionMargin=0.01,
                useSelfCollision=1,
                frictionCoeff=0.5,
                useFaceContact=1,
            )
            self.collapsible_platform = np.insert(
                self.collapsible_platform, index, regenerated_body, axis=0
            )
            if self.textureId:
                env.pybullet_client.changeVisualShape(
                    regenerated_body, -1, textureUniqueId=self.textureId
                )
            elif self.color == "O":
                env.pybullet_client.changeVisualShape(
                    regenerated_body, -1, rgbaColor=[1, 0.5, 0, 1], flags=0
                )
            elif self.color == "R":
                env.pybullet_client.changeVisualShape(
                    regenerated_body, -1, rgbaColor=[1, 0.25, 0, 1], flags=0
                )
            # Anchor Soft Body at the bottom 4 corners
            # ground anchor on vertices 4,5,6,7.
            env.pybullet_client.createSoftBodyAnchor(regenerated_body, 4, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(regenerated_body, 5, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(regenerated_body, 6, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(regenerated_body, 7, -1, -1)

    def _generate(
        self, env, case=1, sElasticStiffness=8, sDampingStiffness=1, texture=None
    ):
        # env.pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        # env.pybullet_client.configureDebugVisualizer(
        #     env.pybullet_client.COV_ENABLE_RENDERING, 0)
        platform = []
        damping_platform = []
        collapsible_platform = []
        damping_tile_pos = []
        collapsible_tile_pos = []
        self.sElasticStiffness = sElasticStiffness
        self.sDampingStiffness = sDampingStiffness

        if (sElasticStiffness + sDampingStiffness) <= 41:
            self.color = "R"
            # collapsibility = 0.0
        elif 41 < (sElasticStiffness + sDampingStiffness) <= 179:
            self.color = "O"
            # collapsibility = 0.50

        if texture:
            self.textureId = env.pybullet_client.loadTexture(
                "blind_walking/envs/env_modifiers/collapsible_platform/%s.png" % texture
            )

        if case == 1:  # front right feet soft platform
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [-0.125, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [-0.125, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.125, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.125, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.375, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.365, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, 0.135, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.875, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.885, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [1.125, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [1.125, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )

            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [-0.5, -0.25, 0.0],
                    globalScaling=1.0,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [-0.5, 0.25, 0.0],
                    globalScaling=1.0,
                    useMaximalCoordinates=True,
                )
            )

            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.250, -0.50, 0.250],
                    globalScaling=0.5,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [1, -0.50, 0.250],
                    globalScaling=0.5,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.250, 0.50, 0.250],
                    globalScaling=0.5,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [1, 0.50, 0.250],
                    globalScaling=0.5,
                    useMaximalCoordinates=True,
                )
            )

            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, 0.625, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, 0.375, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, -0.625, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, -0.385, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )

            # Damping Platform
            damping_platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_tile.urdf",
                    [0.625, -0.125, 0.3875],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            damping_tile_pos.append([0.625, -0.125, 0.3875])
            collapsible_platform.append(
                env.pybullet_client.loadSoftBody(
                    "cube.obj",
                    basePosition=[0.625, -0.125, 0.150],
                    scale=0.25,
                    mass=1.0,
                    useNeoHookean=0,
                    useBendingSprings=1,
                    useMassSpring=1,
                    springElasticStiffness=sElasticStiffness,
                    springDampingStiffness=sDampingStiffness,
                    springDampingAllDirections=1,
                    collisionMargin=0.01,
                    useSelfCollision=1,
                    frictionCoeff=0.5,
                    useFaceContact=1,
                )
            )
            collapsible_tile_pos.append([0.625, -0.125, 0.150])

        if case == 2:  # front left feet soft platform
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [-0.125, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [-0.125, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.125, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.125, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.365, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.375, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, -0.135, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.885, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.875, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [1.125, 0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [1.125, -0.125, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )

            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [-0.5, -0.25, 0.0],
                    globalScaling=1.0,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [-0.5, 0.25, 0.0],
                    globalScaling=1.0,
                    useMaximalCoordinates=True,
                )
            )

            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.250, -0.50, 0.250],
                    globalScaling=0.5,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [1, -0.50, 0.250],
                    globalScaling=0.5,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.250, 0.50, 0.250],
                    globalScaling=0.5,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [1, 0.50, 0.250],
                    globalScaling=0.5,
                    useMaximalCoordinates=True,
                )
            )

            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, 0.625, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, 0.385, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, -0.625, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_platform.urdf",
                    [0.625, -0.375, 0.375],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )

            # Damping Platform
            damping_platform.append(
                env.pybullet_client.loadURDF(
                    "blind_walking/envs/env_modifiers/collapsible_platform/cube_tile.urdf",
                    [0.625, 0.125, 0.3875],
                    globalScaling=0.25,
                    useMaximalCoordinates=True,
                )
            )
            damping_tile_pos.append([0.625, 0.125, 0.3875])
            collapsible_platform.append(
                env.pybullet_client.loadSoftBody(
                    "cube.obj",
                    basePosition=[0.625, 0.125, 0.150],
                    scale=0.25,
                    mass=1.0,
                    useNeoHookean=0,
                    useBendingSprings=1,
                    useMassSpring=1,
                    springElasticStiffness=sElasticStiffness,
                    springDampingStiffness=sDampingStiffness,
                    springDampingAllDirections=1,
                    collisionMargin=0.01,
                    useSelfCollision=1,
                    frictionCoeff=0.5,
                    useFaceContact=1,
                )
            )
            collapsible_tile_pos.append([0.625, 0.125, 0.150])

        # Assign color
        if case in [1, 2]:
            for index, blockId in enumerate(damping_platform):
                if texture:
                    env.pybullet_client.changeVisualShape(
                        blockId, -1, textureUniqueId=self.textureId
                    )
                    env.pybullet_client.changeDynamics(blockId, -1, mass=0.07)
                elif self.color == "O":
                    env.pybullet_client.changeVisualShape(
                        blockId, -1, rgbaColor=[1, 0.5, 0, 1], flags=0
                    )
                    env.pybullet_client.changeDynamics(blockId, -1, mass=0.25)
                elif self.color == "R":
                    env.pybullet_client.changeVisualShape(
                        blockId, -1, rgbaColor=[1, 0.25, 0, 1], flags=0
                    )
                    env.pybullet_client.changeDynamics(blockId, -1, mass=0.07)
            for index, blockId in enumerate(collapsible_platform):
                if texture:
                    env.pybullet_client.changeVisualShape(
                        blockId, -1, textureUniqueId=self.textureId
                    )
                elif self.color == "O":
                    env.pybullet_client.changeVisualShape(
                        blockId, -1, rgbaColor=[1, 0.5, 0, 1], flags=0
                    )
                elif self.color == "R":
                    env.pybullet_client.changeVisualShape(
                        blockId, -1, rgbaColor=[1, 0.25, 0, 1], flags=0
                    )
                # Anchor Soft Body at the bottom 4 corners
                env.pybullet_client.createSoftBodyAnchor(
                    blockId, 4, -1, -1
                )  # ground anchor on vertices 4,5,6,7.
                env.pybullet_client.createSoftBodyAnchor(blockId, 5, -1, -1)
                env.pybullet_client.createSoftBodyAnchor(blockId, 6, -1, -1)
                env.pybullet_client.createSoftBodyAnchor(blockId, 7, -1, -1)
        print("TERRAIN TYPE: Collapsible Platform - Case {}".format(case))

        env.pybullet_client.configureDebugVisualizer(
            env.pybullet_client.COV_ENABLE_RENDERING, 1
        )

        self.platform = platform
        self.damping_platform = damping_platform
        self.collapsible_platform = collapsible_platform
        self.damping_tile_pos = damping_tile_pos
        self.collapsible_tile_pos = collapsible_tile_pos
        # return platform # list of platform id
        # For case 1 & 2: the last id is the soft platform, the 2nd last is the platform on the soft tile)
