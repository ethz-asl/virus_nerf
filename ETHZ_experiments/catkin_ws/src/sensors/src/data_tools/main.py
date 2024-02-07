import os

from pose_sync import PoseSync
from time_sync import TimeSync
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
        return_meas=[
            "/USS1",
            "/USS3",
            "/TOF1",
            "/TOF3",
        ],
    )
    # meass_dict["/CAM1/aligned_depth_to_color/image_raw"] = None
    # meass_dict["/CAM3/aligned_depth_to_color/image_raw"] = None
    
    return meass_dict, times_dict

def main():
    
    data_dir = "/home/spadmin/catkin_ws_ngp/data/office_2"
    bag_name = "office_2_2.bag"
    poses_name = "poses_lidar.csv"
    
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
    
    # synchronized pose topics
    ps = PoseSync(
        data_dir=data_dir,
        bag_name=bag_name,
        poses_name=poses_name,
    )
    write_topics, write_msgs = ps(
        times_dict=times_dict,
    )
    
    # synchronized time topics
    ts = TimeSync(
        data_dir=data_dir,
        bag_name=bag_name,
    )
    replace_topics_r, replace_topics_w, replace_times_r, replace_times_w = ts(
        meass_dict=meass_dict,
        times_dict=times_dict,
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