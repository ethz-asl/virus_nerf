#!/usr/bin/env python
import rospy

from pcl_tools.pcl_publisher import PCLMeasPublisher


def main():
    pub_pcl = PCLMeasPublisher(
        sensor_id=rospy.get_param("sensor_id"),
        pub_frame_id=rospy.get_param("pub_frame_id"),
        use_balm_poses=rospy.get_param("use_balm_poses"),
        data_dir=rospy.get_param("data_dir"),
    )
    pub_pcl.subscribe()

if __name__ == '__main__':
    main()