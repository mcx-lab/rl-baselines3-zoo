#!/usr/bin/env python3

import sys
import os
import rospy
from std_msgs.msg import String

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quadruped_ros.msg import Observation


if __name__ == "__main__":
    try:
        rospy.init_node("quadruped_simulation", anonymous=True)
        pub = rospy.Publisher("observations", Observation, queue_size=1)
        rate = rospy.Rate(33)  # hz
        while not rospy.is_shutdown():
            rospy.loginfo(f"hello world! {rospy.get_time()}")
            msg = Observation()
            msg.data = [1.0] * 92
            pub.publish(msg)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
