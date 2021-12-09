#!/usr/bin/env python3

import rospy
from std_msgs.msg import String


def pub_quadsim():
    pub = rospy.Publisher('observations', String, queue_size=100)
    rospy.init_node('quadruped_simulation', anonymous=True)
    rate = rospy.Rate(33) # hz
    while not rospy.is_shutdown():
        #action = np.zeros(12)
        #obs, reward, done, infos = env.step(action)
        #pub.publish(obs)
        rospy.loginfo(f'hello world! {rospy.get_time()}')   
        pub.publish('hello world!')
        rate.sleep()


if __name__ == "__main__":
    try:
        pub_quadsim()
    except rospy.ROSInterruptException:
        pass
