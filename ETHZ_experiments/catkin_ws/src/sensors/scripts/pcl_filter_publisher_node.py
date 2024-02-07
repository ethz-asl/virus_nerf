#!/usr/bin/env python
import rospy

from pcl_tools.pcl_publisher import PCLFilterPublisher


def main():
    
    if rospy.has_param('lims_x'):
        lims_x = rospy.get_param('lims_x')
    else:
        lims_x = None
    if rospy.has_param('lims_y'):
        lims_y = rospy.get_param('lims_y')
    else:
        lims_y = None
    if rospy.has_param('lims_z'):
        lims_z = rospy.get_param('lims_z')
    else:
        lims_z = None
    if rospy.has_param('lims_r'):
        lims_r = rospy.get_param('lims_r')
    else:
        lims_r = None
    if rospy.has_param('lims_t'):
        lims_t = rospy.get_param('lims_t')
    else:
        lims_t = None
    if rospy.has_param('lims_p'):
        lims_p = rospy.get_param('lims_p')
    else:
        lims_p = None
    if rospy.has_param('offset_depth'):
        offset_depth = rospy.get_param('offset_depth')
    else:
        offset_depth = None
        
    
    pub_pcl = PCLFilterPublisher(
        sub_topic=rospy.get_param("sub_topic"),
        pub_topic=rospy.get_param("pub_topic"),
        lims_x=lims_x,
        lims_y=lims_y,
        lims_z=lims_z,
        lims_r=lims_r,
        lims_t=lims_t,
        lims_p=lims_p,
        offset_depth=offset_depth,
    )
    pub_pcl.subscribe()

if __name__ == '__main__':
    main()