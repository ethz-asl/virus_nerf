#!/usr/bin/env python
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
    
    
class LogRealSense():
    def __init__(
        self,
    ):
        """
        Synchronize time stamps of CAM1 and CAM3
        """
        self.prev_stamp = {
            "status": False,
            "seq": None,
            "secs": None,
            "nsecs": None,
        }
        
        # ROS
        self.topic_cam1 = "/CAM1/color/image_raw"
        self.topic_cam3 = "/CAM3/color/image_raw"
        self.subscribe_cam1 = None
        self.subscribe_cam1 = None
        self.pub_cam1 = rospy.Publisher('/sync/CAM1/image_raw', Image, queue_size=10)
        self.pub_cam3 = rospy.Publisher('/sync/CAM3/image_raw', Image, queue_size=10)
        rospy.init_node('rs_log', anonymous=True)
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        self.subscribe_cam1 = rospy.Subscriber(self.topic_cam1, Image, self._cbCam1)
        self.subscribe_cam3 = rospy.Subscriber(self.topic_cam3, Image, self._cbCam3)
        
        rospy.loginfo(f"LogRealSense.subscribe: Subscribed to: {self.topic_cam1} and {self.topic_cam3}")
        rospy.spin()

    def _cbCam1(
        self,
        data:Image,
    ):
        """
        Callback for topic.
        Args:
            data: data from RealSense; Image
        """
        rospy.loginfo("Enter CAM1")
        
        # log time stamp        
        self.prev_stamp["status"] = True
        self.prev_stamp["seq"] = data.header.seq
        self.prev_stamp["secs"] = data.header.stamp.secs
        self.prev_stamp["nsecs"] = data.header.stamp.nsecs

        # publish image and unsubscribe
        self.pub_cam1.publish(data)
        self.subscribe_cam1.unregister()
        rospy.loginfo("Unsubscribe from CAM1")
            
    def _cbCam3(
        self,
        data:Image,
    ):
        """
        Callback for topic.
        Args:
            data: data from RealSense; Image
        """
        rospy.loginfo("Enter CAM3")
        
        # sync. time stamp
        if self.prev_stamp["status"]:
            data.header.seq = self.prev_stamp["seq"]
            data.header.stamp.secs = self.prev_stamp["secs"]
            data.header.stamp.nsecs = self.prev_stamp["nsecs"]

            # publish image and unsubscribe
            self.pub_cam3.publish(data)
            self.subscribe_cam3.unregister()
            rospy.loginfo("Unsubscribe from CAM3")
        
    

def main():
    log = LogRealSense()
    log.subscribe()

if __name__ == '__main__':
    main()