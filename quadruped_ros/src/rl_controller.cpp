#include <thread>
#include <signal.h>
#include <string.h>
#include "ros/ros.h"
#include "hardware_interface.h"

double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos*(1-rate) + targetPos*rate;
    return p;
}

class RLController : public HardwareInterface
{

public:
    RLController (ros::NodeHandle &RL_nh, const double &loop_rate)
        :HardwareInterface(LeggedType::A1, LOWLEVEL, loop_rate),
        nh_(RL_nh)
    // ~RLController(){};
    {
    long long iter = 0;
    int rate_count = 0;
    double rate = 0.0;
    int loop_iter = 0;
    int is_shutdown = 0;

    // Controller initialisations
    float Kp[3] = {0};  
    float Kd[3] = {0};
    Kp[0] = 10; Kp[1] = 10; Kp[2] = 10;
    Kd[0] = 1.2; Kd[1] = 0.8; Kd[2] = 0.8;

    Mat3<float> kpMat << 50, 0, 0, 0, 50, 0, 0, 0, 50;
    Mat3<float> kdMat << 2, 0, 0, 0, 2, 0, 0, 0, 2;

    float jpos_time = 3.0;
    float wait_time = 3.0;

    static double standPos[12] = {
                         0.0, -0.67, 1.3,
                        -0.0, -0.67, 1.3, 
                         0.0, -0.67, 1.3, 
                        -0.0, -0.67, 1.3
                        };

    static double sitPos[12] = {
                        -0.026550, -1.262121, 2.557070,
                         0.026550, -1.262121, 2.557070,
                        -0.026550, -1.262121, 2.557070,
                         0.026550, -1.262121, 2.557070
                        };

    float action[12] = {0};

    // ROS initialisations
    quadruped_ros::BaseVelocitySensor base_vel_msg;
    sensor_msgs::Imu imu_msg;
    sensor_msgs::JointState motors_msg;
    quadruped_ros::TargetPositionSensor target_pos_msg;
    quadruped_ros::HeightmapSensor heightmap_msg;

    // Initialise publisher and subscriber
    action_sub = nh_.subscribe<quadruped_ros::QuadrupedLegPos> ( "actions", 10, &RL_Controller::actionCallBack, this );
    basevelocity_pub = nh_.advertise<quadruped_ros::BaseVelocitySensor> ( "obs_basevelocity", 3 );
    imu_pub = nh_.advertise<sensor_msgs::Imu> ( "obs_imu", 3 );
    motors_pub = nh_.advertise<sensor_msgs::JointState> ( "obs_motors", 3 );
    targetpos_pub = nh_.advertise<quadruped_ros::TargetPositionSensor> ( "obs_targetpos", 3 );
    heightmap_pub = nh_.advertise<quadruped_ros::HeightmapSensor> ( "obs_heightmap", 3 );
    }

private:
    // receive values from trained RL policy
    void actionCallback( const quadruped_ros::QuadrupedLegPos::ConstPtr &msg) {
        received_action = *msg
        action[12] = {received_action.fr.hip.pos,
                    received_action.fr.upper.pos,
                    received_action.fr.lower.pos,
                    received_action.fl.hip.pos,
                    received_action.fl.upper.pos,
                    received_action.fl.lower.pos,
                    received_action.br.hip.pos,
                    received_action.br.upper.pos,
                    received_action.br.lower.pos,
                    received_action.bl.hip.pos,
                    received_action.bl.upper.pos,
                    received_action.bl.lower.pos
                    };
    }

    void pubBaseVel() { // State estimator->vel
        base_vel_msg.vx = 0.0
        base_vel_msg.vy = 0.0
        basevelocity_pub.publish(base_vel_msg);
    }

    void pubImu() {
        imu_msg.linear_acceleration.x = data.imu.accelerometer[0];
        imu_msg.linear_acceleration.y = data.imu.accelerometer[1];
        imu_msg.linear_acceleration.z = data.imu.accelerometer[2];
        imu_msg.angular_velocity.x = data.imu.gyroscope[0];
        imu_msg.angular_velocity.y = data.imu.gyroscope[1];
        imu_msg.angular_velocity.z = data.imu.gyroscope[2];
        imu_pub.publish(imu_msg);
    }

    void pubMotor() {
        motors_msg.position.clear();
        motors_msg.velocity.clear();
        for (int jidx=0; jidx<12; jidx++)
        {
            motors_msg.position.push_back(data.motorState[jidx].q); 
        }
        for (int jidx=0; jidx<12; jidx++)
        {
            motors_msg.velocity.push_back(data.motorState[jidx].dq); 
        }
        motors_pub.publish(motors_msg);
    }

    // for testing
    void pubTargetPos() {
        // test values
        target_pos_msg.dx = 10.0
        target_pos_msg.dy = 0.0
        targetpos_pub.publish(target_pos_msg);
    }

    // for testing, actual heightmap comes from another module
    void pubHeightmap() {
        // mock values
        heightmap_msg.data.clear();
        for (int pts=0; pts<10; pts++)
        {
            heightmap_msg.data.push_back(0.0); 
        }
        heightmap_pub.publish(heightmap_msg);
    }

    void RobotControl()
    {   
        iter++;

        pubBaseVel(); // need state estimator values
        pubImu();
        pubMotor();
        pubTargetPos(); // test values
        pubHeightmap(); // mock values 

        ////////////////////////////////////
        /*              FSM               */
        // stand up
        if(iter < (500 * jpos_time + 500 * wait_time)) 
        {
            for (int leg = 0; leg < 4; leg++) 
            {
                // _legController->commands[leg].kpJoint = kpMat;
                // _legController->commands[leg].kdJoint = kdMat;
                cmd.motorCmd[leg*3+0].Kp = kpMat(0, 0);
                cmd.motorCmd[leg*3+1].Kp = kpMat(1, 1);
                cmd.motorCmd[leg*3+2].Kp = kpMat(2, 2);

                cmd.motorCmd[leg*3+0].Kd = kdMat(0, 0);
                cmd.motorCmd[leg*3+1].Kd = kdMat(1, 1);
                cmd.motorCmd[leg*3+2].Kd = kdMat(2, 2);
            }
            rate_count++;
            rate = rate_count/200.0;
            for (int leg=0; leg<4; leg++)
                for (int jidx=0; jidx<3; jidx++)
                {
                    // _legController->commands[leg].tauFeedForward[jidx] = 0.;
                    // _legController->commands[leg].qDes[jidx] = standPos[3*leg+jidx];
                    // _legController->commands[leg].qdDes[jidx] = 0.;
                    
                    qDes = jointLinearInterpolation(data.motorState[3*leg+jidx].q, standPos[3*leg+jidx], rate);

                    cmd.motorCmd[leg*3+jidx].tau = 0.;
                    cmd.motorCmd[leg*3+jidx].q = qDes; // standPos[3*leg+jidx];
                    cmd.motorCmd[leg*3+jidx].qd = 0.;
                }
        }
        // sit down
        else if (is_shutdown == 1) { 
            /* make robot enter stand pos before sitting down 
            for better stability? */
            rate_count++;
            rate = rate_count/200.0;
            for (int leg = 0; leg < 4; leg++) 
            {
                cmd.motorCmd[leg*3+0].Kp = kpMat(0, 0);
                cmd.motorCmd[leg*3+1].Kp = kpMat(1, 1);
                cmd.motorCmd[leg*3+2].Kp = kpMat(2, 2);

                cmd.motorCmd[leg*3+0].Kd = kdMat(0, 0);
                cmd.motorCmd[leg*3+1].Kd = kdMat(1, 1);
                cmd.motorCmd[leg*3+2].Kd = kdMat(2, 2);
            }
            for (int leg=0; leg<4; leg++)
                for (int jidx=0; jidx<3; jidx++)
                {
                    qDes = jointLinearInterpolation(data.motorState[3*leg+jidx].q, sitPos[3 * leg + jidx], rate);

                    cmd.motorCmd[leg*3+jidx].tau = 0.;
                    cmd.motorCmd[leg*3+jidx].q = qDes; // sitPos[3*leg+jidx];
                    cmd.motorCmd[leg*3+jidx].qd = 0.;
                }
        }
        // RL controller
        else {
            loop_iter++;
            for (int leg=0; leg<4; leg++)
                for (int jidx=0; jidx<3; jidx++)
                {
                    cmd.motorCmd[leg*3+jidx].q = action[leg*3+jidx];
                    cmd.motorCmd[leg*3+jidx].dq = 0.;
                    cmd.motorCmd[leg*3+jidx].Kp = Kp[jidx];
                    cmd.motorCmd[leg*3+jidx].Kd = Kd[jidx];
                    cmd.motorCmd[leg*3+jidx].tau = 0.;
                }
        }
        /*              FSM               */
        ////////////////////////////////////

        // gravity compensation
        // cmd.motorCmd[FR_0].tau = -0.65f;
        // cmd.motorCmd[FL_0].tau = +0.65f;
        // cmd.motorCmd[RR_0].tau = -0.65f;
        // cmd.motorCmd[RL_0].tau = +0.65f;

    }
};

ros::NodeHandle* nh = nullptr;
std::shared_ptr<RLController> rlController;

void signal_callback_handler (int signum)
{
    printf("Interrupted with SIGINT: %d", signum);
    rlController->rate_count = 0;
    rlController->is_shutdown = 1;
    sleep(rlController->jpos_time + rlController->wait_time);
    rlController->Stop();
    ros::shutdown();
}

int main ()
{
    // Initialise ROS node
    ros::init(argc, argv, "RL_locomotion_ctrl");
    nh = new ros::NodeHandle();

    /* Register signal and signal handler */
    signal ( SIGINT, signal_callback_handler ); // put behind, otherwise will be overwirtten by others such as ROS

    rlController = std::make_shared<RLController> ( 500, nh );
    
    // Initialise hardware interface for A1
    rlController->Init();

    std::thread comm_thread (&RLController::Communicate, rlController);
    std::thread control_thread (&RLController::Control, rlController);
    ros::spin();

    comm_thread.join();
    control_thread.join();

    printf("Stopping Controller \n");
    delete nh;

    return 0;
}