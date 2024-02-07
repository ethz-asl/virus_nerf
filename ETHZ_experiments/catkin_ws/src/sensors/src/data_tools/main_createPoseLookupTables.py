import numpy as np
import pandas as pd
import os
import copy

from pcl_tools.pcl_transformer import PCLTransformer
from pcl_tools.pcl_coordinator import PCLCoordinator


def createLookupTables(
    data_dir:str,
):
    """
    Create lookup table for dynamic transforms.
    Args:
        data_dir: path to data directory; str
    """
    poses_name_list = [
        "poses_lidar_sync1.csv",
        "poses_lidar_sync3.csv",
        "poses_lidar_balm_sync1.csv",
        "poses_lidar_balm_sync3.csv",
    ]
    
    for poses_name in poses_name_list:
        lookupTable(
            data_dir=data_dir,
            poses_name=poses_name,
        )   
    
def lookupTable(
    data_dir:str,
    poses_name:str,
):
    """
    Create lookup table for dynamic transforms.
    Args:
        data_dir: path to data directory; str
        poses_name: name of poses file; str
    """
    if "1" in poses_name:
        stack_id = 1
    elif "3" in poses_name:
        stack_id = 3
    else:
        print("ERROR: main_createPoseLookupTables.py: lookupTable(): stack_id not found.")
    
    # read robot poses
    df_poses_robot = pd.read_csv(
        os.path.join(data_dir, 'poses', poses_name),
        dtype=np.float64,
    )
    
    df_poses_cam = pd.DataFrame(
        columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
        dtype=np.float64,
    )
    
    coord = PCLCoordinator(
        source="CAM"+str(stack_id),
        target="robot",
    )
    
    for i in range(len(df_poses_robot)):
        row = df_poses_robot.iloc[i]
        
        # lookup transform from robot to map
        robot_map = PCLTransformer(
            q=row[['qx', 'qy', 'qz', 'qw']].values,
            t=row[['x', 'y', 'z']].values,
        )
        
        # calculate transform from CAM to robot
        cam_robot = copy.deepcopy(coord.transform)
        cam_map = cam_robot.concatTransform(
            add_transform=robot_map,
            apply_first_add_transform=False,
        )
        
        # add transform from CAM to map to dataframe
        q, t = cam_map.getTransform(type="quaternion")
        time = np.array([row['time']])
        df_poses_cam = df_poses_cam.append(
            pd.DataFrame(
                data=np.concatenate((time, t, q)).reshape(1,-1),
                columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
            ),
            ignore_index=True,
        )
    
    # save lookup table
    df_poses_cam.to_csv(
        path_or_buf=os.path.join(data_dir, 'poses', poses_name.replace('lidar', 'cam')),
        index=False,
    )
    
def main():
    data_dir = "/home/spadmin/catkin_ws_ngp/data/medium_scan_2"
    createLookupTables(
        data_dir=data_dir,
    )
    
if __name__ == '__main__':
    main()