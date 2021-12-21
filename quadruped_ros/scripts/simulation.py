#!/usr/bin/env python3

import os
# import tf
import sys
import rospy
import rospkg
import threading
import pybullet as p
import pybullet_data
import numpy as np
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quadruped_ros.msg import (
    QuadrupedLegPos,
    QuadrupedLeg,
    IMUSensor,
    BaseVelocitySensor,
    TargetPositionSensor,
    HeightmapSensor,
)


_ctrl_actions = [0.0] * 12

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
ACTION_REPEAT = 30


class WalkingSimulation(object):
    def __init__(self):
        self.get_last_vel = [0] * 3
        self.robot_height = 0.30
        self.motor_id_list = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
        self.init_new_pos = [0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.__init_ros()
        self.__load_controller()
        self.__init_simulator()

        add_thread = threading.Thread(target=self.__thread_job)
        add_thread.start()

        if self.camera:
            add_thread_1 = threading.Thread(target=self.__camera_update)
            add_thread_1.start()

    def __init_ros(self):
        self.terrain = rospy.get_param('/simulation/terrain')
        self.camera = rospy.get_param('/simulation/camera')
        self.lateralFriction = rospy.get_param('/simulation/lateralFriction')
        self.spinningFriction = rospy.get_param('/simulation/spinningFriction')
        self.freq = rospy.get_param('/simulation/freq')
        rospy.loginfo("lateralFriction = " + str(self.lateralFriction) +
                      " spinningFriction = " + str(self.spinningFriction))
        rospy.loginfo(" freq = " + str(self.freq))

        # self.robot_tf = tf.TransformBroadcaster()

    def __load_controller(self):
        return

    def __init_simulator(self):
        robot_start_pos = [0, 0, 0.32]
        p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setTimeStep(1.0/self.freq)
        p.setGravity(0, 0, -9.81)
        self.reset = p.addUserDebugParameter("reset", 1, 0, 0)
        p.resetDebugVisualizerCamera(0.2, 45, -30, [1, -1, 1])

        heightPerturbationRange = 0.06
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        if self.terrain == "plain":
            planeShape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
            ground_id = p.createMultiBody(0, planeShape)
            p.resetBasePositionAndOrientation(ground_id, [0, 0, 0], [0, 0, 0, 1])
            p.changeDynamics(ground_id, -1, lateralFriction=self.lateralFriction)

        self.boxId = p.loadURDF("a1/a1.urdf", robot_start_pos, [0, 0, 0, 1], flags=p.URDF_USE_SELF_COLLISION, useFixedBase=False)
        p.changeDynamics(self.boxId, 5, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.boxId, 10, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.boxId, 15, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.boxId, 20, spinningFriction=self.spinningFriction)

        self.__reset_robot()

    def __reset_robot(self):
        robot_z = self.robot_height
        p.resetBasePositionAndOrientation(self.boxId, [0, 0, robot_z], [0, 0, 0, 1])
        p.resetBaseVelocity(self.boxId, [0, 0, 0], [0, 0, 0])
        for j in range(12):
            p.resetJointState(self.boxId, self.motor_id_list[j], self.init_new_pos[j], self.init_new_pos[j+12])
        for j in range(16):
            p.setJointMotorControl2(self.boxId, j, p.VELOCITY_CONTROL, force=0)

    def __thread_job(self):
        rospy.spin()

    def __camera_update(self):
        rate_1 = rospy.Rate(20)
        near = 0.1
        far = 1000
        step_index = 4
        pixelWidth = int(320 / step_index)
        pixelHeight = int(240 / step_index)
        cameraEyePosition = [0.3, 0, 0.26436384367425125]
        cameraTargetPosition = [1.0, 0, 0]
        cameraUpVector = [45, 45, 0]
        self.pointcloud_publisher = rospy.Publisher("/generated_pc", PointCloud2, queue_size=10)
        self.image_publisher = rospy.Publisher("/cam0/image_raw", Image, queue_size=10)

        while not rospy.is_shutdown():
            cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
            get_matrix = p.getMatrixFromQuaternion(cubeOrn)

            T1 = numpy.mat([[0, -1.0/2.0, numpy.sqrt(3.0)/2.0, 0.25], [-1, 0, 0, 0],
                            [0, -numpy.sqrt(3.0)/2.0, -1.0/2.0, 0], [0, 0, 0, 1]])

            T2 = numpy.mat([[get_matrix[0], get_matrix[1], get_matrix[2], cubePos[0]],
                            [get_matrix[3], get_matrix[4], get_matrix[5], cubePos[1]],
                            [get_matrix[6], get_matrix[7], get_matrix[8], cubePos[2]],
                            [0, 0, 0, 1]])

            T3 = numpy.array(T2*T1)

            cameraEyePosition[0] = T3[0][3]
            cameraEyePosition[1] = T3[1][3]
            cameraEyePosition[2] = T3[2][3]
            cameraTargetPosition = (numpy.mat(T3)*numpy.array([[0],[0],[1],[1]]))[0:3]

            q = pyquaternion.Quaternion(matrix=T3)
            cameraQuat = [q[1], q[2], q[3], q[0]]

            # self.robot_tf.sendTransform(self.__fill_tf_message("world", "robot", cubePos, cubeOrn))
            # self.robot_tf.sendTransform(
            #     self.__fill_tf_message("world", "cam", cameraEyePosition, cameraQuat))
            # self.robot_tf.sendTransform(
            #     self.__fill_tf_message("world", "tar", cameraTargetPosition, cubeOrn))

            cameraUpVector = [0, 0, 1]
            viewMatrix = p.computeViewMatrix(
                cameraEyePosition, cameraTargetPosition, cameraUpVector)
            aspect = float(pixelWidth) / float(pixelHeight)
            projectionMatrix = p.computeProjectionMatrixFOV(60, aspect, near, far)
            width, height, rgbImg, depthImg, _ = p.getCameraImage(
                    pixelWidth,
                    pixelHeight,
                    viewMatrix=viewMatrix,
                    projectionMatrix=projectionMatrix,
                    shadow=1,
                    lightDirection=[1, 1, 1],
                    renderer=p.ER_BULLET_HARDWARE_OPENGL)

            # point cloud mehted
            pc_list = []
            pcl_data = pcl.PointCloud()
            fx = (pixelWidth*projectionMatrix[0]) / 2.0
            fy = (pixelHeight*projectionMatrix[5]) / 2.0
            cx = (1-projectionMatrix[2]) * pixelWidth / 2.0
            cy = (1+projectionMatrix[6]) * pixelHeight / 2.0
            cloud_point = [0] * pixelWidth * pixelHeight * 3
            depthBuffer = numpy.reshape(depthImg, [pixelHeight, pixelWidth])
            depth = depthBuffer
            for h in range(0, pixelHeight):
                for w in range(0, pixelWidth):
                    depth[h][w] = float(depthBuffer[h, w])
                    depth[h][w] = far * near / (far - (far - near) * depthBuffer[h][w])
                    Z = float(depth[h][w])
                    if (Z > 4 or Z < 0.01):
                        continue
                    X = (w - cx) * Z / fx
                    Y = (h - cy) * Z / fy
                    XYZ_ = numpy.mat([[X], [Y], [Z], [1]])
                    XYZ = numpy.array(T3*XYZ_)
                    X = float(XYZ[0])
                    Y = float(XYZ[1])
                    Z = float(XYZ[2])
                    cloud_point[h * pixelWidth * 3 + w * 3 + 0] = float(X)
                    cloud_point[h * pixelWidth * 3 + w * 3 + 1] = float(Y)
                    cloud_point[h * pixelWidth * 3 + w * 3 + 2] = float(Z)
                    pc_list.append([X, Y, Z])

            pcl_data.from_list(pc_list)
            pub_pointcloud = PointCloud2()
            pub_pointcloud.header.stamp = rospy.Time().now()
            pub_pointcloud.header.frame_id = "body"
            pub_pointcloud.height = 1
            pub_pointcloud.width = len(pc_list)
            pub_pointcloud.point_step = 12
            pub_pointcloud.fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)]
            pub_pointcloud.data = numpy.asarray(pc_list, numpy.float32).tostring()
            self.pointcloud_publisher.publish(pub_pointcloud)

            # grey image
            pub_image = Image()
            pub_image.header.stamp = rospy.Time().now()
            pub_image.header.frame_id = "cam"
            pub_image.width = width
            pub_image.height = height
            pub_image.encoding = "mono8"
            pub_image.step = width
            grey = pil.fromarray(rgbImg)
            pub_image.data = numpy.asarray(grey.convert('L')).reshape([1,-1]).tolist()[0]
            self.image_publisher.publish(pub_image)

            rate_1.sleep()

    def run(self):
        rate = rospy.Rate(self.freq)  # Hz
        reset_flag = p.readUserDebugParameter(self.reset)
        while not rospy.is_shutdown():
            # check reset button state
            if(reset_flag < p.readUserDebugParameter(self.reset)):
                reset_flag = p.readUserDebugParameter(self.reset)
                rospy.logwarn("reset the robot")
                self.__reset_robot()

            self.__simulation_step()

            rate.sleep()

    def __simulation_step(self):
        # get data from simulator
        observations = self.__get_data_from_sim()

        # pub msg
        self.__pub_base_velocity_msg(observations["base_velocity"])
        self.__pub_imu_msg(observations["imu"])
        self.__pub_joint_states(observations["joint_states"])
        self.__pub_target_position(observations["target_position"])
        self.__pub_heightmap(observations["heightmap"])

        global _ctrl_actions
        for i in range(ACTION_REPEAT):
            # clip actions
            clipped_actions = self._clip_actions(_ctrl_actions)
            # apply actions
            self._apply_actions(clipped_actions)
            # step simulation
            p.stepSimulation()

    def _clip_actions(self, motor_commands):
        max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        current_motor_angles, _, _, _ = self.__get_motor_joint_states(self.boxId)
        motor_commands = np.clip(
            motor_commands,
            np.array(current_motor_angles) - max_angle_change,
            np.array(current_motor_angles) + max_angle_change,
        )
        return motor_commands

    def _apply_actions(self, motor_commands):
        kp = np.array([100.0] * 12)
        kd = np.array([1.0, 2.0, 2.0] * 4)
        motor_angles, motor_vels, _, _ = self.__get_motor_joint_states(self.boxId)
        motor_torques = kp * (motor_commands - motor_angles) - kd * motor_vels
        p.setJointMotorControlArray(
            bodyIndex=self.boxId,
            jointIndices=self.motor_id_list,
            controlMode=p.TORQUE_CONTROL,
            forces=motor_torques,
        )

    def __get_data_from_sim(self):
        get_matrix = []
        get_velocity = []
        get_invert = []

        pose_orn = p.getBasePositionAndOrientation(self.boxId)
        get_velocity = p.getBaseVelocity(self.boxId)
        get_invert = p.invertTransform(pose_orn[0], pose_orn[1])
        get_matrix = p.getMatrixFromQuaternion(get_invert[1])

        observations = {}

        imu_data = [0] * 10
        leg_data = {}
        leg_data["state"] = [0] * 24
        leg_data["name"] = [""] * 12

        # IMU data
        imu_data[3] = pose_orn[1][0]
        imu_data[4] = pose_orn[1][1]
        imu_data[5] = pose_orn[1][2]
        imu_data[6] = pose_orn[1][3]

        imu_data[7] = get_matrix[0] * get_velocity[1][0] + get_matrix[1] * \
            get_velocity[1][1] + get_matrix[2] * get_velocity[1][2]
        imu_data[8] = get_matrix[3] * get_velocity[1][0] + get_matrix[4] * \
            get_velocity[1][1] + get_matrix[5] * get_velocity[1][2]
        imu_data[9] = get_matrix[6] * get_velocity[1][0] + get_matrix[7] * \
            get_velocity[1][1] + get_matrix[8] * get_velocity[1][2]
        # calculate the acceleration of the robot
        linear_X = (get_velocity[0][0] - self.get_last_vel[0]) * self.freq
        linear_Y = (get_velocity[0][1] - self.get_last_vel[1]) * self.freq
        linear_Z = 9.8 + (get_velocity[0][2] - self.get_last_vel[2]) * self.freq
        imu_data[0] = get_matrix[0] * linear_X + \
            get_matrix[1] * linear_Y + get_matrix[2] * linear_Z
        imu_data[1] = get_matrix[3] * linear_X + \
            get_matrix[4] * linear_Y + get_matrix[5] * linear_Z
        imu_data[2] = get_matrix[6] * linear_X + \
            get_matrix[7] * linear_Y + get_matrix[8] * linear_Z
        observations["imu"] = imu_data

        # joint motors data
        joint_positions, joint_velocities, _, joint_names = self.__get_motor_joint_states(self.boxId)
        leg_data["state"][0:12] = joint_positions
        leg_data["state"][12:24] = joint_velocities
        leg_data["name"] = joint_names
        observations["joint_states"] = leg_data

        # target position data
        max_distance = 0.02
        dy_target = 0 - pose_orn[0][1]
        dy_target = max(min(dy_target, max_distance / 2), -max_distance / 2)
        dx_target = np.sqrt(pow(max_distance, 2) - pow(dy_target, 2))
        def to_local_frame(dx, dy, yaw):
            dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
            dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
            return dx_local, dy_local
        dx_target_local, dy_target_local = to_local_frame(dx_target, dy_target, pose_orn[1][2])
        observations["target_position"] = [dx_target_local, dy_target_local]

        # CoM velocity
        self.get_last_vel = [get_velocity[0][0], get_velocity[0][1], get_velocity[0][2]]
        observations["base_velocity"] = self.get_last_vel

        # heightmap data
        observations["heightmap"] = [0] * 46

        return observations

    def __get_motor_joint_states(self, robot):
        joint_number_range = range(p.getNumJoints(robot))
        joint_states = p.getJointStates(robot, joint_number_range)
        joint_infos = [p.getJointInfo(robot, i) for i in joint_number_range]
        joint_states, joint_name = \
            zip(*[(j, i[1]) for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED])
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques, joint_name

    def __pub_base_velocity_msg(self, bv_data):
        pub_bv = rospy.Publisher("obs_basevelocity", BaseVelocitySensor, queue_size=30)
        bv_msg = BaseVelocitySensor()
        bv_msg.vx = bv_data[0]
        bv_msg.vy = bv_data[1]
        bv_msg.vz = bv_data[2]
        bv_msg.header.stamp = rospy.Time.now()
        bv_msg.header.frame_id = "body"
        pub_bv.publish(bv_msg)

    def __pub_imu_msg(self, imu_data):
        pub_imu = rospy.Publisher("obs_imu", Imu, queue_size=30)
        imu_msg = Imu()
        imu_msg.linear_acceleration.x = imu_data[0]
        imu_msg.linear_acceleration.y = imu_data[1]
        imu_msg.linear_acceleration.z = imu_data[2]
        imu_msg.angular_velocity.x = imu_data[7]
        imu_msg.angular_velocity.y = imu_data[8]
        imu_msg.angular_velocity.z = imu_data[9]
        imu_msg.orientation.x = imu_data[3]
        imu_msg.orientation.y = imu_data[4]
        imu_msg.orientation.z = imu_data[5]
        imu_msg.orientation.w = imu_data[6]
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "body"
        pub_imu.publish(imu_msg)

    def __pub_joint_states(self, joint_states):
        pub_js = rospy.Publisher("obs_motors", JointState, queue_size=30)
        js_msg = JointState()
        js_msg.name = []
        js_msg.position = []
        js_msg.velocity = []
        i = 0
        for _ in joint_states["name"]:
            js_msg.name.append(joint_states["name"][i].decode('utf-8'))
            js_msg.position.append(joint_states["state"][i])
            js_msg.velocity.append(joint_states["state"][12+i])
            i += 1
        js_msg.header.stamp = rospy.Time.now()
        js_msg.header.frame_id = "body"
        pub_js.publish(js_msg)

    def __pub_target_position(self, tp_data):
        pub_tp = rospy.Publisher("obs_targetpos", TargetPositionSensor, queue_size=30)
        tp_msg = TargetPositionSensor()
        tp_msg.dx = tp_data[0]
        tp_msg.dy = tp_data[1]
        tp_msg.header.stamp = rospy.Time.now()
        tp_msg.header.frame_id = "body"
        pub_tp.publish(tp_msg)

    def __pub_heightmap(self, hm_data):
        pub_hm = rospy.Publisher("obs_heightmap", HeightmapSensor, queue_size=30)
        hm_msg = HeightmapSensor()
        hm_msg.data = hm_data
        hm_msg.header.stamp = rospy.Time.now()
        hm_msg.header.frame_id = "body"
        pub_hm.publish(hm_msg)


def callback_action(obs):
    global _ctrl_actions
    _ctrl_actions[0] = obs.fr.hip.pos
    _ctrl_actions[1] = obs.fr.upper.pos
    _ctrl_actions[2] = obs.fr.lower.pos
    _ctrl_actions[3] = obs.fl.hip.pos
    _ctrl_actions[4] = obs.fl.upper.pos
    _ctrl_actions[5] = obs.fl.lower.pos
    _ctrl_actions[6] = obs.br.hip.pos
    _ctrl_actions[7] = obs.br.upper.pos
    _ctrl_actions[8] = obs.br.lower.pos
    _ctrl_actions[9] = obs.bl.hip.pos
    _ctrl_actions[10] = obs.bl.upper.pos
    _ctrl_actions[11] = obs.bl.lower.pos


if __name__ == '__main__':
    rospy.init_node('quadruped_simulator', anonymous=True)
    rospy.Subscriber("actions", QuadrupedLegPos, callback_action)
    walking_simulation = WalkingSimulation()
    walking_simulation.run()
