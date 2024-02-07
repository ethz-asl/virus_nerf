#!/usr/bin/env python
import rospy

from pcl_tools.pcl_publisher import PCLStaticPublisher


def main():
    pub_pcl = PCLStaticPublisher(
        pub_topic=rospy.get_param("pub_topic"),
        pub_freq=rospy.get_param("pub_freq"),
        data_dir=rospy.get_param("data_dir"),
        map_name=rospy.get_param("map_name"),
    )

if __name__ == '__main__':
    main()