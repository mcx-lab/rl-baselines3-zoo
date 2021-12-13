#!/usr/bin/env python3

import rospy
from std_msgs.msg import String


if __name__ == "__main__":
    try:
        rospy.init_node("quadruped_simulation", anonymous=True)
        pub = rospy.Publisher("observations", String, queue_size=100)
        rate = rospy.Rate(33)  # hz
        while not rospy.is_shutdown():
            rospy.loginfo(f"hello world! {rospy.get_time()}")
            msg = String()
            msg.data = "hello world!"
            pub.publish(msg)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
