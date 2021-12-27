#pragma once

#include <sys/timerfd.h>
#include <sys/mman.h>
#include <cmath>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>

#include <vector>
#include <fstream>
#include <string.h>

#include "unitree_legged_sdk/unitree_legged_sdk.h"

// Headers for joystick
#include "joystick_comm.h"
#include <sys/ioctl.h>
#include <linux/joystick.h>
#include <fcntl.h>
#define JOY_DEV "/dev/input/js0"

#define DEBUG //Un-comment for Debugging

using namespace UNITREE_LEGGED_SDK;

class HardwareInterface
{

public:
    HardwareInterface( const LeggedType& type = LeggedType::A1, const uint8_t& level = LOWLEVEL, const double &loop_rate = 500.0 )
     : safe(type), udp(level)
    {
        if( loop_rate>0 ){
            dt_ = 1.0/loop_rate;
        } else{
            std::cout << "Loop rate should be more than zero! Setting to default 500Hz" << std::endl;
            dt_ = 1.0/500.0;
        }

        joy_dt_ = 0.02;
    }
    virtual ~HardwareInterface(){};

    void Init();
    void Stop();
    void Communicate();
    void Control();
    void RC_Interface();

protected:
    virtual void RobotControl() = 0;
    LowCmd cmd = {0};
    LowState data = {0};
    JoystickCmd joy_cmd = {0};
    double dt_;
    double joy_dt_;
    bool joyPresent;
    bool running_;

private:
    void InitError( const char* reason, bool printErrno );
    void PrefaultStack();
    void SetupScheduler();
    void SetInitValue();
    void InitJoy();
    void getJoyCommands();

    Safety safe;
    UDP udp;

    bool comm_ready_;

    LowCmd cmd_;
    LowState data_;
        
    LowCmd cmd_buf_;
    LowState data_buf_;
    JoystickCmd joy_cmd_buf_;

    pthread_mutex_t cmd_mut_;
    pthread_mutex_t data_mut_;
    pthread_mutex_t joy_cmd_mut_;

    // Joystick Params
    struct js_event js; 
	int joy_fd;
	int *axis=NULL, num_of_axis=0, num_of_buttons=0, x;
	char *button=NULL, name_of_joystick[80];

};
