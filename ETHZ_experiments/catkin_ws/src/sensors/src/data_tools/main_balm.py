import numpy as np
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.getcwd(), '..', 'pcl_tools'))
from pcl_loader import PCLLoader
from pcl_transformer import PCLTransformer
from pcl_merger import PCLMerger


def main_createDataset():
    """
    Create dataset used for BALM
    """
    data_dir = "/home/spadmin/catkin_ws_ngp/data/medium_scan_2"
    lidar_pcl_dir = "lidars/filtered"
    lidar_poses_name = "poses/poses_lidar.csv"
    balm_dir = "balm"
    num_scans = 300
    
    if not os.path.exists(os.path.join(data_dir, balm_dir)):
        os.makedirs(os.path.join(data_dir, balm_dir))
    
    # Load poses and times
    pcl_merger = PCLMerger(
        data_dir=data_dir,
    )
    poses_kiss, times_kiss = pcl_merger.loadPoses(
        pose_format="vector",
        filename=lidar_poses_name,
    ) # (N, 7), (N,)
    times_lidar, filenames_lidar = pcl_merger.getTimes(
        pcl_dir=lidar_pcl_dir,
    ) # (M,), (M,)

    # Match poses and times
    idxs = pcl_merger.matchTimes(
        times_subset=times_kiss,
        times=times_lidar,
    ) # (N,)
    filenames_match = np.array(filenames_lidar)[idxs] # (N,)
    
    # subsample data to have desired number of scans
    idxs = np.linspace(0, len(filenames_match)-1, num_scans, dtype=np.int32)
    filenames_match = filenames_match[idxs]
    poses_kiss = poses_kiss[idxs]
    times_kiss = times_kiss[idxs]
    
    # Convert poses
    poses_balm = np.zeros((poses_kiss.shape[0], 4, 4))
    for i, pose in enumerate(poses_kiss):
        T = PCLTransformer(
            t=pose[:3],
            q=pose[3:],
        ).getTransform(
            type="matrix",
        )
        poses_balm[i] = T
        
    # Save poses and pcd files
    pcl_merger.savePoses(
        poses=poses_balm,
        times=times_kiss,
        filename=os.path.join(balm_dir, "alidarPose.csv"),
        pose_format="matrix_times",
    )
    for i, fname in enumerate(filenames_match):
        shutil.copy(
            src=os.path.join(data_dir, lidar_pcl_dir, fname),
            dst=os.path.join(data_dir, balm_dir, f"full{i}.pcd"),
        ) 
            
def main_iteratePoses():
    pcl_loader = PCLLoader(
        data_dir="/home/spadmin/catkin_ws_ngp/data/office_2",
    )
    
    _, times = pcl_loader.loadPoses(
        pose_format="matrix_times",
        filename="balm/alidarPose.csv",
    )
    poses = pcl_loader.loadPoses(
        pose_format="matrix",
        filename="balm/poses_lidar_balm.csv",
    )
        
    pcl_loader.savePoses(
        poses=poses,
        times=times,
        filename="balm/alidarPose.csv",
        pose_format="matrix_times",
    )
    
if __name__ == '__main__':
    main_createDataset()
    # main_iteratePoses()