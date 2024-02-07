#!/usr/bin/env python

from rosbag import Bag
import rospy
import numpy as np
import pandas as pd
import cv2 as cv
from cv_bridge import CvBridge
import copy
import os
from nav_msgs.msg import Odometry
from tqdm import tqdm



class RosbagWrapper():
    def __init__(
        self,
        data_dir:str,
        bag_name:str,
    ) -> None:
        self.data_dir = data_dir
        self.bag_path = os.path.join(data_dir, bag_name)
        
        self.cv_bridge = CvBridge()
        
        self.max_depth_csv_cols = 10
        self.tof_cols = None
        self.depth_cols = None
    
    def read(
        self,
        return_time:list=[],
        return_meas:list=[],
        save_meas:list=[],
    ):
        """
        Get messages from a particular topic.
        Args:
            return_time: return time stamps; list of str
            return_meas: return measurements; list of str
            save_meas: save measurements; list of str
        Returns:
            meass: measurements from each topic; list of np.array of floats (N)
            times: times of measurements in seconds from each topic; list np.array of floats (N)
        """
        # unique topics
        topics = return_time + return_meas + save_meas
        topics = np.unique(topics)
        print(f"INFO: RosbagWrapper.read: topics={topics} from {self.bag_path}")

        # create dataframes for saving measurements
        df_meas = {}
        img_counters = {}
        for topic in save_meas:
            if "USS" in topic:
                df_meas[topic] = pd.DataFrame(columns=['time', 'meas'])
            elif "TOF" in topic:
                df_meas[topic] = pd.DataFrame(columns=self._getToFColumns())
            elif "depth" in topic:
                img_counters[topic] = 0
            elif "color" in topic:
                img_counters[topic] = 0
        
        meass_dict = { topic:[] for topic in return_meas }
        times_dict = { topic:[] for topic in return_time }
        with Bag(self.bag_path, 'r') as bag:
            
            for topic_temp, msg_temp, _ in tqdm(bag):
                
                if not topic_temp in topics:
                    continue
                
                if "USS" in topic_temp:
                    meas, time, df_meas = self._readMsgUss(
                        topic_temp=topic_temp,
                        msg_temp=msg_temp,
                        return_msg=topic_temp in return_meas,
                        return_time=topic_temp in return_time,
                        save_meas=topic_temp in save_meas,
                        df_meas=df_meas,
                    )
                elif "TOF" in topic_temp:
                    meas, stds, time, df_meas = self._readMsgTof(
                        topic_temp=topic_temp,
                        msg_temp=msg_temp,
                        return_msg=topic_temp in return_meas,
                        return_time=topic_temp in return_time,
                        save_meas=topic_temp in save_meas,
                        df_meas=df_meas,
                    )
                elif "depth" in topic_temp:
                    img, time, img_counters, df_meas = self._readMsgImgDepth(
                        topic_temp=topic_temp,
                        msg_temp=msg_temp,
                        return_msg=topic_temp in return_meas,
                        return_time=topic_temp in return_time,
                        save_meas=topic_temp in save_meas,
                        img_counters=img_counters,
                        df_meas=df_meas,
                    )
                elif "color" in topic_temp:
                    img, time, img_counter = self._readMsgImgColor(
                        topic_temp=topic_temp,
                        msg_temp=msg_temp,
                        return_msg=topic_temp in return_meas,
                        return_time=topic_temp in return_time,
                        save_meas=topic_temp in save_meas,
                        img_counters=img_counters,
                    )

                if topic_temp in return_meas:
                    meass_dict[topic_temp].append(meas)
                if topic_temp in return_time:
                    times_dict[topic_temp].append(time)

        if not os.path.exists(os.path.join(self.data_dir, "measurements")):
                os.makedirs(os.path.join(self.data_dir,  "measurements"))
        for topic in df_meas.keys():
            df_meas[topic].to_csv(os.path.join(self.data_dir, "measurements", f"{topic[1:].replace('/', '_')}.csv"), index=False)
                
        meass_dict_arr = {}
        for topic in return_meas:
            meass_dict_arr[topic] = np.array(meass_dict[topic])
            
        times_dict_arr = {}
        for topic in return_time:
            times_dict_arr[topic] = np.array(times_dict[topic])
        return meass_dict_arr, times_dict_arr
    
    def _readMsgUss(
        self,
        topic_temp:str,
        msg_temp,
        return_msg:bool,
        return_time:bool,
        save_meas:bool,
        df_meas:pd.DataFrame,
    ):
        if save_meas:
            df_meas[topic_temp] = df_meas[topic_temp].append(
                pd.DataFrame(
                    data=np.array([[
                        msg_temp.header.stamp.to_sec(),
                        msg_temp.meas,
                    ]]),
                    columns=['time', 'meas'],
                ),
                ignore_index=True,
            )
        
        meas = None
        if return_msg:
            meas = msg_temp.meas 
            
        time = None
        if return_time:
            time = msg_temp.header.stamp.to_sec()
        return meas, time, df_meas
    
    def _readMsgTof(
        self,
        topic_temp:str,
        msg_temp,
        return_msg:bool,
        return_time:bool,
        save_meas:bool,
        df_meas:dict,
    ):
        if save_meas:
            df_meas[topic_temp] = df_meas[topic_temp].append(
                pd.DataFrame(
                    data=np.hstack([
                        np.array([msg_temp.header.stamp.to_sec()]).reshape(1, -1),
                        np.array(msg_temp.meas).reshape(1, -1),
                        np.array(msg_temp.stds).reshape(1, -1),
                    ]),
                    columns=self._getToFColumns(),
                ),
                ignore_index=True,
            )
        meas, stds = None, None
        if return_msg:
            meas = msg_temp.meas
            stds = msg_temp.stds
            
        time = None
        if return_time:
            time = msg_temp.header.stamp.to_sec()
        return meas, stds, time, df_meas
    
    def _readMsgImgDepth(
        self,
        topic_temp:str,
        msg_temp,
        return_msg:bool,
        return_time:bool,
        save_meas:bool,
        img_counters:dict,
        df_meas:dict,
    ):
        if save_meas or return_msg:
            img = self.cv_bridge.imgmsg_to_cv2(msg_temp, desired_encoding="passthrough")
            
        if save_meas:
            folder_name = f"{topic_temp[1:].replace('/', '_')}"
            img_name = f"img{img_counters[topic_temp]}.npy"
                
            if not os.path.exists(os.path.join(self.data_dir, "measurements", folder_name)):
                os.makedirs(os.path.join(self.data_dir,  "measurements", folder_name))
                
            np.save(
                file=os.path.join(self.data_dir,  "measurements", folder_name, img_name), 
                arr=img.astype(np.uint16),
            )
            img_counters[topic_temp] += 1
            
        if not return_msg:
            img = None
            
        time = None
        if return_time:
            time = msg_temp.header.stamp.to_sec()
        return img, time, img_counters, df_meas
    
    def _readMsgImgColor(
        self,
        topic_temp:str,
        msg_temp,
        return_msg:bool,
        return_time:bool,
        save_meas:bool,
        img_counters:dict,
    ):
        if save_meas or return_msg:
            img = self.cv_bridge.imgmsg_to_cv2(msg_temp, desired_encoding=msg_temp.encoding)
            
        if save_meas:
            folder_name = f"{topic_temp[1:].replace('/', '_')}"
            img_name = f"img{img_counters[topic_temp]}.png"
                
            if not os.path.exists(os.path.join(self.data_dir, "measurements", folder_name)):
                os.makedirs(os.path.join(self.data_dir,  "measurements", folder_name))
                
            cv.imwrite(
                filename=os.path.join(self.data_dir,  "measurements", folder_name, img_name), 
                img=img,
            )
            img_counters[topic_temp] += 1
            
        if not return_msg:
            img = None
            
        time = None
        if return_time:
            time = msg_temp.header.stamp.to_sec()
        return img, time, img_counters
    
    def _getToFColumns(
        self,
    ):
        if self.tof_cols != None:
            return self.tof_cols
        
        col_meas = [f"meas_{i}" for i in range(0, 64)]
        col_stds = [f"stds_{i}" for i in range(0, 64)]
        
        self.tof_cols = ['time'] + col_meas + col_stds
        return self.tof_cols
    
    def newBag(
        self,
        new_bag_path:str,
        keep_topics:list=[],
        write_topics:list=[],
        write_msgs:list=[],
        replace_topics_r:list=[],
        replace_topics_w:list=[],
        replace_times_r:list=[],
        replace_times_w:list=[],
    ):
        """
        Create new bag file, write messages to a particular topics and replace time stamps of a particular topics.
        Args:
            new_bag_path: path to bag file to write to; str
            keep_topics: list of topics to keep; list of str
            write_topics: list of topics to write to; list of str
            write_msgs: list of messages to write; list of list of rosbag messages
            replace_topics_r: list of topics to read from; list of str
            replace_topics_w: list of topics to write to; list of str
            replace_times_r: list of times of msgs to read from; list of np.array of floats (N)
            replace_times_w: list of times of msgs to write to; list of np.array of floats (N)
        """
        with Bag(new_bag_path, 'w') as bag:
            
            if len(keep_topics) > 0:
                print(f"INFO: RosbagWrapper.newBag: keep_topics={keep_topics}")
                self._keep(
                    bag=bag,
                    topics=keep_topics,
                )
            
            if len(write_topics) > 0:
                print(f"INFO: RosbagWrapper.newBag: write_topics={write_topics}")
                for i, topic in tqdm(enumerate(write_topics)):
                    self._write(
                        bag=bag,
                        topic=topic,
                        msgs=write_msgs[i],
                    )
            
            if len(replace_topics_r) > 0:
                print(f"INFO: RosbagWrapper.newBag: replace_topics_r={replace_topics_r}")
                for i, topic in tqdm(enumerate(replace_topics_r)):
                    self._replaceTimeStamp(
                        bag=bag,
                        topic_read=topic,
                        topic_write=replace_topics_w[i],
                        time_read=replace_times_r[i],
                        time_write=replace_times_w[i],
                    )
                
    def _keep(
        self,
        bag:Bag,
        topics:list,
    ):
        """
        Keep messages from a particular topics.
        Args:
            bag: opened rosbag object to write to; rosbag.Bag
            topics: topics to keep; list of str
        """
        for topic, msg, _ in tqdm(Bag(self.bag_path).read_messages()):
            if topic in topics:
                bag.write(topic, msg, msg.header.stamp)
    
    def _write(
        self,
        bag:Bag,
        topic:str,
        msgs:list,
    ):
        """
        Get messages from a particular topic.
        Args:
            bag: opened rosbag object to write to; rosbag.Bag
            topic: topic name; str
            msgs: messages to write; list of rosbag messages
        """ 
        for msg in tqdm(msgs):
            bag.write(topic, msg, msg.header.stamp)
            
    def _replaceTimeStamp(
        self,
        bag:Bag,
        topic_read:str,
        topic_write:str,
        time_read:np.array,
        time_write:np.array,
    ):
        """
        Replace the time stamp of topic_read with time_write.
        Args:
            bag: opened rosbag object to write to; rosbag.Bag
            topic_read: topic name to read from; str
            topic_write: topic name to write to; str
            time_read: time of msg to read from; np.array of floats (N)
            time_write: time of msg to write to; np.array of floats (N)
        """
        counter = 0
        for topic_temp, msg_temp, _ in tqdm(Bag(self.bag_path).read_messages()):
            
            if topic_temp != topic_read:
                continue
            
            # if "USS" in topic_read:
            #     print(f"msg_temp={msg_temp.header.stamp.to_sec()},      time_read[counter]={time_read[counter]}")
            
            # enter while loop and save message if this source message corresponds to a target message
            # enter multiple times the while loop if this source message corresponds to multiple target messages
            msg_temp_sec = msg_temp.header.stamp.to_sec()
            while msg_temp_sec == time_read[counter]:
                time_write_ros = rospy.Time.from_sec(time_write[counter])
                msg_temp.header.stamp = time_write_ros
                bag.write(topic_write, msg_temp, time_write_ros)
            
                counter += 1
                if counter >= len(time_write):
                    break
            
            if counter >= len(time_write):
                break
                
        if counter != len(time_write):
            print(f"ERROR: RosbagWrapper.writeTimeStamp: counter={counter} != len(time_sync)={len(time_write)}")
    
    def writeTimeStamp(
        self,
        bag_path_sync:str,
        topic_async:str,
        topic_sync:str,
        time_async:np.array,
        time_sync:np.array,
    ):
        """
        Write messages to a particular topic.
        Args:
            bag_path_sync: path to bag file to write to; str
            topic_async: topic name to copy data from; str
            topic_sync: topic name to write data to; str
            time_async: time of msg to copy data from; np.array of floats (N)
            time_sync: synchronized time; np.array of floats (N)
        """
        print(f"topic_async={topic_async}")
        
        with Bag(bag_path_sync, 'w') as bag:
            
            counter = 0
            for b_topic, b_msg, b_time in Bag(self.bag_path).read_messages():
                
                if b_topic != topic_async:
                    continue
                
                # enter while loop and save message if this source message corresponds to a target message
                # enter multiple times the while loop if this source message corresponds to multiple target messages
                b_msg_time = b_msg.header.stamp.to_sec()
                while b_msg_time == time_async[counter]:
                    ros_time = rospy.Time.from_sec(time_sync[counter])
                    b_msg.header.stamp = ros_time
                    bag.write(topic_sync, b_msg, ros_time)
                
                    counter += 1
                    if counter >= len(time_sync):
                        break
                
                if counter >= len(time_sync):
                    break
                
        if counter != len(time_sync):
            print(f"ERROR: RosbagWrapper.writeTimeStamp: counter={counter} != len(time_sync)={len(time_sync)}")
            
    def cropBag(
        self,
        bag_path_to_crop:str,
        msgs_range:tuple,
    ):
        """
        Write messages to a particular topic.
        Args:
            bag_path_to_crop: path to bag file that should be croped; str
            msgs_range: number of messages to copy; tuple of ints (start, end)
        """
        with Bag(self.bag_path, 'w') as bag:
            
            counter = 0
            for b_topic, b_msg, b_time in Bag(bag_path_to_crop).read_messages():
                
                counter += 1
                if counter < msgs_range[0]:
                    continue
                if counter >= msgs_range[1]:
                    break
                    
                bag.write(b_topic, b_msg, b_time)
                
        print(f"INFO: RosbagWrapper.cropBag: {counter-msgs_range[0]} messages written to {self.bag_path}")
                
    def merge(
        self,
        bag_path_out:str,
        bag_paths_ins:list,
        keep_topics:list=None,
        delete_ins:bool=False,
    ):
        """
        Merge multiple bag files into one.
        Args:
            bag_path_out: path to bag file to write to; str
            bag_paths_ins: list of paths to bag files to read from; list of str
            keep_topics: list of list that indicates for each bag file which topics to read,
                         if topic list is equal to 'all', all topics are merged of this list; list of list of str
            delete_ins: list that indicates to delete input bag; list of bool
        """
        if keep_topics is None:
            keep_topics = ["all"] * len(bag_paths_ins)
        
        bag_path_out_origianl = copy.copy(bag_path_out)
        if bag_path_out in bag_paths_ins:
            bag_path_out = bag_path_out.replace(".bag", "_this_is_a_random_string.bag")

        with Bag(bag_path_out, 'w') as o: 
            
            for i, ifile in enumerate(bag_paths_ins):
                with Bag(ifile, 'r') as ib:
                    
                    for topic, msg, t in ib:
                        if (keep_topics[i] == "all") or (topic in keep_topics[i]):
                            o.write(topic, msg, t)
                        
        if bag_path_out != bag_path_out_origianl:
            os.rename(bag_path_out, bag_path_out_origianl)
        
        
        for i, ifile in enumerate(bag_paths_ins):
            if delete_ins[i]:
                os.remove(ifile)
                
    def poses2msgs(
        self,
        poses:np.array,
        times:np.array,
        frame_id:str,
    ):
        """
        Convert poses to ROS messages.
        Args:
            poses: poses; np.array of floats (N,7)
            times: times of poses in seconds; np.array of floats (N)
            frame_id: frame id of poses; str
        Returns:
            msgs: ROS messages; list of Odometry messages
        """
        msgs = []
        for i in range(len(poses)):
            msg = Odometry()
            msg.header.stamp = rospy.Time.from_sec(times[i])
            msg.header.frame_id = frame_id
            msg.pose.pose.position.x = poses[i,0]
            msg.pose.pose.position.y = poses[i,1]
            msg.pose.pose.position.z = poses[i,2]
            msg.pose.pose.orientation.x = poses[i,3]
            msg.pose.pose.orientation.y = poses[i,4]
            msg.pose.pose.orientation.z = poses[i,5]
            msg.pose.pose.orientation.w = poses[i,6]
            msgs.append(msg)
        return msgs



        

def main():

    bag_wrap = RosbagWrapper(
        bag_path="/home/spadmin/catkin_ws_ngp/data/DataSync/test.bag",
    )
    bag_wrap.cropBag(
        bag_path_to_crop="/home/spadmin/catkin_ws_ngp/data/DataSync/office_2.bag",
        msgs_range=(150740, 160741),
    )

if __name__ == "__main__":
    main()
    
    
    
# def _getDepthImgColumns(
#         self,
#     ):
#         if self.depth_cols != None:
#             return self.depth_cols

#         mesh_height, mesh_width = np.meshgrid(
#             np.arange(480),
#             np.arange(640),
#             indexing='ij',
#         )
#         mesh_height = mesh_height.flatten()
#         mesh_width = mesh_width.flatten()
#         col_meas = [f"h{mesh_height[i]}_w{mesh_width[i]}" for i in range(len(mesh_height))]
        
#         self.depth_cols = ['time'] + col_meas
#         return self.depth_cols