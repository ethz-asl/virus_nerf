#!/usr/bin/env python
import numpy as np
import rospy

from pcl_tools.pcl_loader import PCLLoader
from pcl_tools.pcl_transformer import PCLTransformer

        
def convertPosesBALM2ETHZ(
    data_dir:str
):
    pcl_loader = PCLLoader(
        data_dir=data_dir,
    )
    
    _, times = pcl_loader.loadPoses(
        pose_format="matrix_times",
        filename="balm/alidarPose.csv",
    )
    balm_poses = pcl_loader.loadPoses(
        pose_format="matrix",
        filename="balm/poses_lidar_balm.csv",
    )
    
    poses = np.zeros((0,7))
    for i in range(times.shape[0]):
        trans = PCLTransformer(
            T=balm_poses[i],
        )
        q, t = trans.getTransform(
            type="quaternion",
        )
        pose = np.hstack((t, q))
        poses = np.concatenate((poses, pose.reshape((1,7))), axis=0)
        
    pcl_loader.savePoses(
        poses=poses,
        times=times,
        filename="poses/poses_lidar_balm.csv",
        pose_format="vector",
    )
    
def shutdown_handler():
    # Code to be executed on shutdown goes here
    balm_dir = rospy.get_param("file_path")
    data_dir = balm_dir.replace("balm/", "")
    convertPosesBALM2ETHZ(
        data_dir=data_dir,
    )

if __name__ == '__main__':
    try:
        rospy.init_node('pose_balm2ethz_node', anonymous=True)
        rospy.on_shutdown(shutdown_handler)    
        rospy.spin()
    except rospy.ROSInterruptException:
        pass