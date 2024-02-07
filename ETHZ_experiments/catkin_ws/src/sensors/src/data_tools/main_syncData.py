import os

from pose_sync import PoseSync
from meas_sync import MeasSync
from rosbag_wrapper import RosbagWrapper

def readData(
    data_dir:str,
    bag_name:str,
):
    """
    Read data in one step for faster processing.
    Args:
        data_dir: path to data directory; str
        bag_name: name of bag file; str
    Returns:
        meass_dict: dictionary with measurements; dict
        times_dict: dictionary with times; dict
    """
    bag_wrapper = RosbagWrapper(
        data_dir=data_dir,
        bag_name=bag_name,
    )
    meass_dict, times_dict = bag_wrapper.read(
        return_time=[
            "/CAM1/color/image_raw",
            "/CAM3/color/image_raw",
            "/CAM1/aligned_depth_to_color/image_raw",
            "/CAM3/aligned_depth_to_color/image_raw",
            "/USS1",
            "/USS3",
            "/TOF1",
            "/TOF3",
        ],
        return_meas=[],
    )
    return meass_dict, times_dict

def main():
    data_dir = "/home/spadmin/catkin_ws_ngp/data/medium_scan_2"
    bag_name = "medium_scan_2.bag"
    poses_name = "poses_lidar.csv"
    poses_balm_name = "poses_lidar_balm.csv"
    
    # topics to copy and paste fom old bag
    keep_topics = [
        "/CAM1/depth/color/points",
        "/CAM3/depth/color/points",
        "/rslidar_points",
    ]
    
    # read data from rosbag
    meass_dict, times_dict = readData(
        data_dir=data_dir,
        bag_name=bag_name,
    )
    
    # synchronized measurements
    ms = MeasSync(
        data_dir=data_dir,
        bag_name=bag_name,
    )
    replace_topics_r, replace_topics_w, replace_times_r, replace_times_w, masks = ms(
        meass_dict=meass_dict,
        times_dict=times_dict,
    )
    
    # synchronized poses
    ps = PoseSync(
        data_dir=data_dir,
        bag_name=bag_name,
        poses_name=poses_name,
    )
    write_topics, write_msgs = ps(
        return_msgs=True,
        times_dict=times_dict,
        masks=masks,
    )
    ps = PoseSync(
        data_dir=data_dir,
        bag_name=bag_name,
        poses_name=poses_balm_name,
    )
    ps(
        return_msgs=False,
        times_dict=times_dict,
        masks=masks,
    )
    

    
    ps.newBag(
        new_bag_path=os.path.join(data_dir, bag_name.replace('.bag', '_sync.bag')),
        keep_topics=keep_topics,
        write_topics=write_topics,
        write_msgs=write_msgs,
        replace_topics_r=replace_topics_r,
        replace_topics_w=replace_topics_w,
        replace_times_r=replace_times_r,
        replace_times_w=replace_times_w,
    )


if __name__ == "__main__":
    main()