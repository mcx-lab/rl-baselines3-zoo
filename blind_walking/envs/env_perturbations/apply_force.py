from turtle import pos
from typing import Tuple
import numpy as np
import csv
import os

def draw_debug_sphere(pybullet_client, position: Tuple[float, float, float], rgba_color=[0, 1, 1, 1], radius=0.02) -> int:
    ballShape = pybullet_client.createCollisionShape(shapeType=pybullet_client.GEOM_SPHERE, radius=radius)
    ball_id = pybullet_client.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=ballShape, basePosition=position, baseOrientation=[0, 0, 0, 1]
    )
    pybullet_client.changeVisualShape(ball_id, -1, rgbaColor=rgba_color)
    pybullet_client.setCollisionFilterGroupMask(ball_id, -1, 0, 0)
    return ball_id

# obstacle_pos = [7.5, 19.5, 31.5, 13.5, 25.5, 37.5]    # Platform obstacle pos
obstacle_pos = [2.5, 6.5, 10.5]         # Short platform obstacle pos
# obstacle_pos = [5.0, 11.0, 19.0, 26.0]                # Hurdle obstacle pos
# obstacle_pos = [5, 10, 15, 20]                        # Heightfield obstacle pos
# obstacle_pos = [7.5, 21.9, 36.3, 14.4, 28.8, 44.7]    # Staircase obstacle pos
obstacle_pos.sort()

log_path = "object_thrown_time.csv"

def clear_thrown_logs():
    if os.path.exists(log_path):
        os.remove(log_path)

def log_object_thrown(time, target_position):
    with open(log_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([time, *target_position])

class ThrowObject:
    def __init__(self):
        self.throw_windows = [(p - 0.4, p + 0.4) for p in obstacle_pos]
        self.throw_directions = [1, 1, 1, 1, 1, 1]
        self.window_idx = 0
        clear_thrown_logs()

    def applyDisturbance(self, pybullet_client, target_position, desired_direction):
        p = pybullet_client
        
        initial_displacement = -desired_direction
        bodyPos = initial_displacement + target_position
        bodyPos[2] += 0.04
        
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
        base_position = np.array(robot.GetBasePosition())

        if self.window_idx >= len(self.throw_windows):
            return

        window_start, window_end = self.throw_windows[self.window_idx]
        if window_start < base_position[0] < window_end:
            print(f"Throwing object at time {time}, position {base_position}")
            pybullet_client = robot.pybullet_client 

            base_position = np.array(robot.GetBasePosition())
            base_position[0] += 0.1
            
            dir = self.throw_directions[self.window_idx]
            desired_direction = np.array([0, 1.0, 0]) if dir == 1 else np.array([0, -1.0, 0])

            self.applyDisturbance(pybullet_client, target_position=base_position, desired_direction=desired_direction)
            log_object_thrown(time, base_position)
            # Move to next window
            self.window_idx += 1
        """
        elif robot._step_counter == 5100:
            print(f"Throwing object at time {time}")
            pybullet_client = robot.pybullet_client 
            quadruped = robot.quadruped

            base_position = np.array(robot.GetBasePosition())
            base_position[0] += 0.1
            desired_direction = np.array([0, -1.0, 0])

            self.applyDisturbance(pybullet_client, target_position=base_position, desired_direction=desired_direction)
        """

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
