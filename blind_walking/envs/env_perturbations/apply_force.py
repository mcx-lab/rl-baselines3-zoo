from turtle import pos
from typing import Tuple
import numpy as np

def draw_debug_sphere(pybullet_client, position: Tuple[float, float, float], rgba_color=[0, 1, 1, 1], radius=0.02) -> int:
    ballShape = pybullet_client.createCollisionShape(shapeType=pybullet_client.GEOM_SPHERE, radius=radius)
    ball_id = pybullet_client.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=ballShape, basePosition=position, baseOrientation=[0, 0, 0, 1]
    )
    pybullet_client.changeVisualShape(ball_id, -1, rgbaColor=rgba_color)
    pybullet_client.setCollisionFilterGroupMask(ball_id, -1, 0, 0)
    return ball_id

class ThrowObject:
    def __init__(self):
        pass 

    def applyDisturbance(self, pybullet_client, target_position, desired_direction):
        p = pybullet_client
        
        initial_displacement = -desired_direction
        bodyPos = initial_displacement + target_position
        
        collisionId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        visualId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                                           rgbaColor=[1, 0.4, 0.1, 1])
        bodyId = p.createMultiBody(baseMass=3, baseCollisionShapeIndex=collisionId, baseVisualShapeIndex=visualId
                                       , basePosition=[10,0,0])

        init_vel = desired_direction * 10
        p.resetBasePositionAndOrientation(bodyUniqueId=bodyId, posObj=bodyPos, ornObj=[0,0,0,1])
        p.resetBaseVelocity(objectUniqueId=bodyId,linearVelocity=init_vel)

        return bodyId

    def apply(self, robot):
        time =  robot.GetTimeSinceReset()
        if robot._step_counter == 100:
            print(f"Throwing object at time {time}")
            pybullet_client = robot.pybullet_client 
            quadruped = robot.quadruped

            base_position = np.array(robot.GetBasePosition())
            desired_direction = np.array([0, 1.0, 0])

            self.applyDisturbance(pybullet_client, target_position=base_position, desired_direction=desired_direction)

        elif robot._step_counter == 2100:
            print(f"Throwing object at time {time}")
            pybullet_client = robot.pybullet_client 
            quadruped = robot.quadruped

            base_position = np.array(robot.GetBasePosition())
            desired_direction = np.array([0, -1.0, 0])

            self.applyDisturbance(pybullet_client, target_position=base_position, desired_direction=desired_direction)


class ApplyForce:

    def __init__(self):
        pass 
        self.ball_ids = []

    def clear_visual_shapes(self):
        for i in self.ball_ids:
            self.pybullet_client.removeBody(i)
        self.ball_ids.clear()

    def apply(self, robot):
        time =  robot.GetTimeSinceReset()
        if 2.0 <= robot.GetTimeSinceReset() <= 2.05:
            print(f"Applying force at time {time}")
            pybullet_client = robot.pybullet_client
            quadruped = robot.quadruped

            self.pybullet_client = pybullet_client
            self.clear_visual_shapes()

            base_position = robot.GetBasePosition()

            force = [0, 50.0, 0]
            
            # Draw a ball to visualize where force is applied
            ball_displacement = [0, 0, 0.2]
            ball_position = [x1 + x2 for x1, x2 in zip(base_position, ball_displacement)]
            ball_id = draw_debug_sphere(pybullet_client, ball_position, radius=0.02)
            self.ball_ids.append(ball_id)
            # Draw a second ball to visualize direction in which force is applied
            ball_displacement = [0, 1.0, 0.2]
            ball_position = [x1 + x2 for x1, x2 in zip(base_position, ball_displacement)]
            ball_id = draw_debug_sphere(pybullet_client, ball_position, radius=0.02)
            self.ball_ids.append(ball_id)

            pybullet_client.applyExternalForce(quadruped, 0, force, [0, 0, 0], pybullet_client.WORLD_FRAME)

        elif robot.GetTimeSinceReset() > 2.05:
            self.clear_visual_shapes()
