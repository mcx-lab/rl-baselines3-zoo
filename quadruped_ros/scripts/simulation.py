#!/usr/bin/env python3

import os
# import tf
import rospy
import rospkg
import ctypes
import threading
import pybullet as p
import pybullet_data
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


class StructPointer(ctypes.Structure):
    _fields_ = [("eff", ctypes.c_double * 12)]


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
        self.stand_kp = rospy.get_param('/simulation/stand_kp')
        self.stand_kd = rospy.get_param('/simulation/stand_kd')
        self.joint_kp = rospy.get_param('/simulation/joint_kp')
        self.joint_kd = rospy.get_param('/simulation/joint_kd')
        rospy.loginfo("lateralFriction = " + str(self.lateralFriction) +
                      " spinningFriction = " + str(self.spinningFriction))
        rospy.loginfo(" freq = " + str(self.freq) + " PID = " +
                      str([self.stand_kp, self.stand_kd, self.joint_kp, self.joint_kd]))

        # self.robot_tf = tf.TransformBroadcaster()

    def __load_controller(self):
        return

    def __init_simulator(self):
        robot_start_pos = [0, 0, 0.42]
        p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.resetSimulation()
        p.setTimeStep(1.0/self.freq)
        p.setGravity(0, 0, -9.81)
        self.reset = p.addUserDebugParameter("reset", 1, 0, 0)
        p.resetDebugVisualizerCamera(0.2, 45, -30, [1, -1, 1])

        heightPerturbationRange = 0.06
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        if self.terrain == "plane":
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
        p.resetBasePositionAndOrientation(
            self.boxId, [0, 0, robot_z], [0, 0, 0, 1])
        p.resetBaseVelocity(self.boxId, [0, 0, 0], [0, 0, 0])
        for j in range(12):
            p.resetJointState(
                self.boxId, self.motor_id_list[j], self.init_new_pos[j], self.init_new_pos[j+12])

        for _ in range(10):
            p.stepSimulation()
            imu_data, leg_data, _ = self.__get_data_from_sim()

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
        imu_data, leg_data, base_pos = self.__get_data_from_sim()

        # pub msg
        self.__pub_nav_msg(base_pos, imu_data)
        self.__pub_imu_msg(imu_data)
        self.__pub_joint_states(leg_data)

        global _ctrl_actions
        tau = _ctrl_actions
        # TODO: nn is using position control
        p.setJointMotorControlArray(bodyUniqueId=self.boxId,
                                    jointIndices=self.motor_id_list,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=tau)

        p.stepSimulation()

    def __get_data_from_sim(self):
        get_matrix = []
        get_velocity = []
        get_invert = []
        imu_data = [0] * 10
        leg_data = {}
        leg_data["state"] = [0] * 24
        leg_data["name"] = [""] * 12

        pose_orn = p.getBasePositionAndOrientation(self.boxId)

        get_velocity = p.getBaseVelocity(self.boxId)
        get_invert = p.invertTransform(pose_orn[0], pose_orn[1])
        get_matrix = p.getMatrixFromQuaternion(get_invert[1])

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

        # joint data
        joint_state = p.getJointStates(boxId, motor_id_list)
        leg_data[0:12] = [joint_state[0][0], joint_state[1][0], joint_state[2][0],
                          joint_state[3][0], joint_state[4][0], joint_state[5][0],
                          joint_state[6][0], joint_state[7][0], joint_state[8][0],
                          joint_state[9][0], joint_state[10][0], joint_state[11][0]]
        leg_data[12:24] = [joint_state[0][1], joint_state[1][1], joint_state[2][1],
                           joint_state[3][1], joint_state[4][1], joint_state[5][1],
                           joint_state[6][1], joint_state[7][1], joint_state[8][1],
                           joint_state[9][1], joint_state[10][1], joint_state[11][1]]

        # CoM velocity
        self.get_last_vel = [get_velocity[0][0], get_velocity[0][1], get_velocity[0][2]]

        return imu_data, leg_data, pose_orn[0]

    def __pub_nav_msg(self, base_pos, imu_data):
        pub_odom = rospy.Publisher("obs_odometry", Odometry, queue_size=30)
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        odom.child_frame_id = "body"
        odom.pose.pose.position.x = base_pos[0]
        odom.pose.pose.position.y = base_pos[1]
        odom.pose.pose.position.z = base_pos[2]
        odom.pose.pose.orientation.x = imu_data[3]
        odom.pose.pose.orientation.y = imu_data[4]
        odom.pose.pose.orientation.z = imu_data[5]
        odom.pose.pose.orientation.w = imu_data[6]
        pub_odom.publish(odom)

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