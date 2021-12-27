#include "hardware_interface.h"

#define MAX_STACK_SIZE 16384
#define TASK_PRIORITY 49

void HardwareInterface::InitError ( const char* reason, bool printErrno )
{
    printf ( "FAILED TO INITIALIZE HARDWARE: %s\n", reason );
    if ( printErrno ) {
        // printf("Error: %s\n", strerror(errno));
    }
    exit ( -1 );
}

void HardwareInterface::PrefaultStack()
{
    printf ( "[Init] Prefault stack...\n" );
    volatile char stack[MAX_STACK_SIZE];
    memset ( const_cast<char*> ( stack ), 0, MAX_STACK_SIZE );
    if ( mlockall ( MCL_CURRENT | MCL_FUTURE ) == -1 ) {
        InitError (
            "mlockall failed.  This is likely because you didn't run robot as "
            "root.\n",
            true );
    }
}

void HardwareInterface::SetupScheduler()
{
    printf ( "[Init] Setup RT Scheduler...\n" );
    struct sched_param params;
    params.sched_priority = TASK_PRIORITY;
    if ( sched_setscheduler ( 0, SCHED_FIFO, &params ) == -1 ) {
        InitError ( "sched_setscheduler failed.\n", true );
    }
}

void HardwareInterface::Init()
{
    running_ = true;
    comm_ready_ = true;

    printf ( "[HardwareBridge] Init stack\n" );
    PrefaultStack();
    printf ( "[HardwareBridge] Init scheduler\n" );
    SetupScheduler();

    udp.InitCmdData(cmd);
    SetInitValue();
    InitJoy();    
}

void HardwareInterface::Stop()
{
    running_ = false;
    close(joy_fd);

}

void HardwareInterface::SetInitValue()
{
    for (uint leg = 0; leg<4; leg++){
        for(uint jindx = 0; jindx<3; jindx++)
        {
            cmd_buf_.motorCmd[leg*3+jindx].q = 0.0;
            cmd_buf_.motorCmd[leg*3+jindx].dq = 0.0;
            cmd_buf_.motorCmd[leg*3+jindx].tau = 0.0;
            cmd_buf_.motorCmd[leg*3+jindx].Kp = 0.0;
            cmd_buf_.motorCmd[leg*3+jindx].Kd = 0.0;
        }
    }

    for (uint leg = 0; leg<4; leg++){
        for(uint jindx = 0; jindx<3; jindx++)
        {
            data_buf_.motorState[leg*3+jindx].q = 0.0;
            data_buf_.motorState[leg*3+jindx].dq = 0.0;
            data_buf_.motorState[leg*3+jindx].tauEst = 0.0;
        }
    }

    for (uint i = 0; i<2; i++){
        joy_cmd_buf_.joyLeftAnalog[i] = 0.0;
        joy_cmd_buf_.joyRightAnalog[i] = 0.0;
    }

}

void HardwareInterface::InitJoy()
{
    joyPresent = true;

    if( ( joy_fd = open( JOY_DEV , O_RDONLY)) == -1 )
    {
        printf( "Couldn't open joystick\n" );
        joyPresent = false;
    } else {
        printf("Joystick detected: %s\n\t%d axis\n\t%d buttons\n\n"
        , name_of_joystick
        , num_of_axis
        , num_of_buttons );
    }
    

    ioctl( joy_fd, JSIOCGAXES, &num_of_axis );
    ioctl( joy_fd, JSIOCGBUTTONS, &num_of_buttons );
    ioctl( joy_fd, JSIOCGNAME(80), &name_of_joystick );

    axis = (int *) calloc( num_of_axis, sizeof( int ) );
    button = (char *) calloc( num_of_buttons, sizeof( char ) );

    fcntl( joy_fd, F_SETFL, O_NONBLOCK ); /* use non-blocking mode */
}

void HardwareInterface::getJoyCommands()
{
    /* read the joystick state */
    read(joy_fd, &js, sizeof(struct js_event));
    
      /* see what to do with the event */
    switch (js.type & ~JS_EVENT_INIT)
    {
      case JS_EVENT_AXIS:
        axis   [ js.number ] = js.value;
        break;
      case JS_EVENT_BUTTON:
        button [ js.number ] = js.value;
        break;
    }

	fflush(stdout); 
}

void HardwareInterface::Communicate()
{
    auto timerFd = timerfd_create(CLOCK_MONOTONIC, 0);
    int seconds = (int)dt_;
    int nanoseconds = (int)( 1e9 * std::fmod(dt_, 1.f ));

    itimerspec timerSpec;
    timerSpec.it_interval.tv_sec = seconds;
    timerSpec.it_value.tv_sec = seconds;
    timerSpec.it_interval.tv_nsec = nanoseconds;
    timerSpec.it_value.tv_nsec = nanoseconds;

    timerfd_settime(timerFd, 0, &timerSpec, nullptr);

    unsigned long long missed = 0;
    static int init_counter = 0;

    while(running_)
    {
        #ifdef DEBUG
            auto start = std::chrono::high_resolution_clock::now();
        #endif

        udp.SetSend(cmd_); 
        udp.GetRecv(data_);

        if (init_counter < 20) {
            init_counter++;
        } else {
            comm_ready_ = true;
        }
        
        pthread_mutex_lock( &cmd_mut_);
            memcpy (&cmd_, &cmd_buf_, sizeof(LowCmd));
        pthread_mutex_unlock(&cmd_mut_);

        if(init_counter >= 20)
        {
            safe.PositionLimit(cmd_);
            safe.PowerProtect(cmd_, data_, 10);
        }

        udp.Send();
        udp.Recv();

        pthread_mutex_lock ( &data_mut_ );
            memcpy ( &data_buf_, &data_, sizeof ( LowState ) );
        pthread_mutex_unlock ( &data_mut_ );

        int m = read ( timerFd, &missed, sizeof(missed));
        (void) m;

        #ifdef DEBUG
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds> ( stop - start );
            // std::cout << "Communication Time: " << duration.count() << " microseconds" << std::endl;
        #endif
    }
}

void HardwareInterface::Control()
{
    auto timerFd = timerfd_create ( CLOCK_MONOTONIC, 0 );
    int seconds = ( int ) dt_;
    int nanoseconds = ( int ) ( 1e9 * std::fmod ( dt_, 1.f ) );

    itimerspec timerSpec;
    timerSpec.it_interval.tv_sec = seconds;
    timerSpec.it_value.tv_sec = seconds;
    timerSpec.it_value.tv_nsec = nanoseconds;
    timerSpec.it_interval.tv_nsec = nanoseconds;

    timerfd_settime ( timerFd, 0, &timerSpec, nullptr );
    unsigned long long missed = 0;

    while ( !comm_ready_ ) {
        usleep ( 100 );
    }

    while ( running_ ) {

        #ifdef DEBUG
            auto start = std::chrono::high_resolution_clock::now();
        #endif

        pthread_mutex_lock ( &data_mut_ );
            memcpy ( &data, &data_buf_, sizeof ( LowState ) );
        pthread_mutex_unlock ( &data_mut_ );

        pthread_mutex_lock ( &joy_cmd_mut_);
            memcpy ( &joy_cmd, &joy_cmd_buf_, sizeof( JoystickCmd ));
        pthread_mutex_unlock ( &joy_cmd_mut_ );

        RobotControl();

        pthread_mutex_lock ( &cmd_mut_ );
            memcpy ( &cmd_buf_, &cmd, sizeof ( LowCmd ) );
        pthread_mutex_unlock ( &cmd_mut_ );

        int m = read ( timerFd, &missed, sizeof ( missed ) ); // wait for loop time, refer to http://manpages.ubuntu.com/manpages/bionic/man2/timerfd_create.2.html
        ( void ) m;

        #ifdef DEBUG
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds> ( stop - start );
            std::cout << "Control Time: " << duration.count() << " microseconds" << std::endl;
            // control_time = std::chrono::duration<long int>(duration).count();
        #endif
    }
}

void HardwareInterface::RC_Interface()
{
    auto timerFd = timerfd_create(CLOCK_MONOTONIC, 0);
    int seconds = (int)joy_dt_;
    int nanoseconds = (int)( 1e9 * std::fmod(joy_dt_, 1.f ));

    itimerspec timerSpec;
    timerSpec.it_interval.tv_sec = seconds;
    timerSpec.it_value.tv_sec = seconds;
    timerSpec.it_interval.tv_nsec = nanoseconds;
    timerSpec.it_value.tv_nsec = nanoseconds;

    timerfd_settime(timerFd, 0, &timerSpec, nullptr);

    unsigned long long missed = 0;
    if(joyPresent) {

        while(running_)
        {
            #ifdef DEBUG
                auto start = std::chrono::high_resolution_clock::now();
            #endif

            getJoyCommands();

            pthread_mutex_lock ( &joy_cmd_mut_ );
                joy_cmd_buf_.joyLeftAnalog[0]  =  0.5 * axis[0]/(32767.0);
                joy_cmd_buf_.joyLeftAnalog[1]  = -0.5 * axis[1]/(32767.0);
                joy_cmd_buf_.joyRightAnalog[0] = 0.5 * axis[3]/(32767.0);
                joy_cmd_buf_.joyRightAnalog[1] = 0.5 * axis[4]/(32767.0);
            pthread_mutex_unlock ( &joy_cmd_mut_ );

            int m = read ( timerFd, &missed, sizeof(missed));
            (void) m;

            #ifdef DEBUG
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds> ( stop - start );
                std::cout << "Communication Time: " << duration.count() << " microseconds" << std::endl;
                std::cout << "cmd_ leg 0: motor 2 " << cmd_.motorCmd[2].q << std::endl;
            #endif
        }
    }
}