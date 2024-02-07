#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from sensors.msg import USS, TOF
from cv_bridge import CvBridge
import numpy as np
import struct
from abc import abstractmethod
import os

import sensor_msgs.point_cloud2 as pc2


from pcl_tools.pcl_processor import PCLProcessor
from pcl_tools.pcl_creator import PCLCreatorUSS, PCLCreatorToF, PCLCreatorRS
from pcl_tools.pcl_coordinator import PCLCoordinator
from pcl_tools.pcl_loader import PCLLoader


class PCLPublisher():
    def __init__(
        self,
        pub_topic:str,
        sub_topic:str,
        sub_topic_msg_type:object,
    ) -> None:
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.sub_topic_msg_type = sub_topic_msg_type
        
        # ROS
        self.sub = None
        self.pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=10)
        rospy.init_node('PCLPublisher', anonymous=True)
        
        # colors
        self.color_floats = {
            "w": struct.unpack('!f', bytes.fromhex('00FFFFFF'))[0],
            "r": struct.unpack('!f', bytes.fromhex('00FF0000'))[0],
            "g": struct.unpack('!f', bytes.fromhex('0000FF00'))[0],
            "b": struct.unpack('!f', bytes.fromhex('000000FF'))[0],
        }
    
    @abstractmethod 
    def _callback(
        self,
        msg:object,
    ):
        pass
    
        
    def subscribe(
        self,
    ):
        """
        Subscribe to topic and wait for data.
        """
        self.subscribe_uss = rospy.Subscriber(self.sub_topic, self.sub_topic_msg_type, self._callback)       
        rospy.loginfo(f"PCLPublisher.subscribe: Subscribed to: {self.sub_topic}")
        
        rospy.spin()
        
    def _publishPCL(
        self,
        xyzi:np.array,
        header:Header=None,
    ):
        """
        Publish pointcloud as Pointcloud2 ROS message.
        Args:
            xyzi: pointcloud [x ,y ,z, intensity]; np.array (N,4)
            header: ROS header to publish; Header
        """
        xyzi = xyzi[(xyzi[:,0]!=np.NAN) & (xyzi[:,1]!=np.NAN) & (xyzi[:,2]!=np.NAN)]
        xyzi = xyzi.astype(dtype=np.float32)
        
        rgb = self.color * np.ones((xyzi.shape[0], 1), dtype=np.float32) # (H, W)
        xyzirgb = np.concatenate((xyzi, rgb), axis=1)
        
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header

        # Define the point fields (attributes)        
        pointcloud_msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('rgb', 16, PointField.UINT32, 1),
        ]
        pointcloud_msg.height = 1
        pointcloud_msg.width = xyzirgb.shape[0]

        # Float occupies 4 bytes. Each point then carries 12 bytes.
        pointcloud_msg.point_step = len(pointcloud_msg.fields) * 4 
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.is_bigendian = False # assumption
        pointcloud_msg.is_dense = True
        
        pointcloud_msg.data = xyzirgb.tobytes()
        self.pub.publish(pointcloud_msg)


class PCLMeasPublisher(PCLPublisher):
    def __init__(
        self,
        sensor_id:str,
        pub_frame_id:str,
        use_balm_poses:bool,
        data_dir:str=None,
    ):
        self.sensor_id = sensor_id
        self.pub_frame_id = pub_frame_id
        
        if "USS" in self.sensor_id:
            msg_type = USS
        elif "TOF" in self.sensor_id:
            msg_type = TOF
        elif "CAM" in self.sensor_id:
            msg_type = Image
        else:
            rospy.logerr(f"PCLPublisher.subscribe: Unknown sensor_id: {self.sensor_id}")
            
        sub_topic = "/" + self.sensor_id
        if "CAM" in self.sensor_id:
            sub_topic += "/aligned_depth_to_color/image_raw"
            self.cv_bridge = CvBridge()
        
        super().__init__(
            pub_topic="/" + self.sensor_id + "_pcl",
            sub_topic=sub_topic,
            sub_topic_msg_type=msg_type,
        )
        
        # sensor type specific variables
        if "USS" in self.sensor_id:
            self.pcl_creator = PCLCreatorUSS()
            self.color = self.color_floats["b"]
        elif "TOF" in self.sensor_id:
            self.pcl_creator = PCLCreatorToF()
            self.color = self.color_floats["w"]
        elif "CAM" in self.sensor_id:
            self.pcl_creator = PCLCreatorRS(
                data_dir=data_dir,
                sensor_id=self.sensor_id,
            )
            self.color = self.color_floats["g"]
        else:
            rospy.logerr(f"PCLPublisher.__init__: Unknown sensor_id: {self.sensor_id}")
        
        # sensor number specific variables
        if "1" in self.sensor_id:
            self.sub_frame_id = "CAM1"
        elif "3" in self.sensor_id:
            self.sub_frame_id = "CAM3"
        else:
            rospy.logerr(f"PCLPublisher.__init__: Unknown sensor_id: {self.sensor_id}")
            
        if self.sub_frame_id != self.pub_frame_id:
            
            lookup_table_path = None
            if self.pub_frame_id == "map":
                if use_balm_poses:
                    lookup_table_path = os.path.join(data_dir, "poses", "poses_cam_balm_sync"+self.sensor_id[-1]+".csv")
                else:
                    lookup_table_path = os.path.join(data_dir, "poses", "poses_cam_sync"+self.sensor_id[-1]+".csv")
                
            self.pcl_coordinator = PCLCoordinator(
                source=self.sub_frame_id,
                target=self.pub_frame_id,
                lookup_table_path=lookup_table_path,
            )
        
    def _callback(
        self,
        msg:object,
    ):
        """
        Callback function for subscriber.
        """
        if "USS" in self.sensor_id or "TOF" in self.sensor_id:
            meas = msg.meas
        elif "CAM" in self.sensor_id:
            meas = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        else:
            rospy.logerr(f"PCLPublisher._callback: Unknown sensor_id: {self.sensor_id}")
            
        xyz = self.pcl_creator.meas2pcl(
            meas=meas,
        )
        
        if self.sub_frame_id != self.pub_frame_id:
            xyz = self.pcl_coordinator.transformCoordinateSystem(
                xyz=xyz,
                time=msg.header.stamp.to_sec(),
            )
            
        header = msg.header
        header.frame_id = self.pub_frame_id  
        xyzi = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=1)
        self._publishPCL(
            xyzi=xyzi,
            header=header,
        )
        
    
class PCLFilterPublisher(PCLPublisher):
    def __init__(
        self,
        sub_topic:str,
        pub_topic:str,
        lims_x:tuple=None,
        lims_y:tuple=None,
        lims_z:tuple=None,
        lims_r:tuple=None,
        lims_t:tuple=None,
        lims_p:tuple=None,
        offset_depth:float=None,
    ):
        self.lims_x = lims_x
        self.lims_y = lims_y
        self.lims_z = lims_z
        self.lims_r = lims_r
        self.lims_t = lims_t
        self.lims_p = lims_p
        self.offset_depth = offset_depth
        
        super().__init__(
            pub_topic=pub_topic,
            sub_topic=sub_topic,
            sub_topic_msg_type=PointCloud2,
        )
        
        self.pcl_processor = PCLProcessor()
        self.color = self.color_floats["w"]
        
    def _callback(
        self,
        msg:PointCloud2,
    ):
        """
        Callback function for subscriber.
        Args:
            msg: ROS pointcloud message; PointCloud2
        """
        # start_time = rospy.get_time()
            
        xyzi = []
        for p in pc2.read_points(msg, field_names = ("x", "y", "z", "intensity"), skip_nans=True):
            xyzi.append(p)
        xyzi = np.array(xyzi)
            
        xyzi = self.pcl_processor.limitXYZ(
            xyzi=xyzi,
            x_lims=self.lims_x,
            y_lims=self.lims_y,
            z_lims=self.lims_z,
        )
        
        xyzi = self.pcl_processor.limitRTP(
            xyzi=xyzi,
            r_lims=self.lims_r,
            t_lims=self.lims_t,
            p_lims=self.lims_p,
        )
        
        xyzi = self.pcl_processor.offsetDepth(
            xyzi=xyzi,
            offset=self.offset_depth,
        )
        
        self._publishPCL(
            xyzi=xyzi,
            header=msg.header,
        )
        
        # stop_time = rospy.get_time()
        # rospy.logwarn(f"PCLFilterPublisher._callback: Time elapsed: {stop_time-start_time}s")
        
        
class PCLStaticPublisher(PCLPublisher):
    def __init__(
        self,
        pub_topic:str,
        pub_freq:float,
        data_dir:str,
        map_name:str,
    ):
        
        super().__init__(
            pub_topic=pub_topic,
            sub_topic=None,
            sub_topic_msg_type=None,
        )
        
        self.pcl_processor = PCLProcessor()
        self.color = self.color_floats["w"]
        
        pcl_loader = PCLLoader(
            data_dir=data_dir,
        )
        xyz = pcl_loader.loadPCL(
            filename=map_name,
        )
        xyzi = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=1)
        
        header = Header()
        header.frame_id = "map"
        
        rate = rospy.Rate(pub_freq)
        while not rospy.is_shutdown():
            self._publishPCL(
                xyzi=xyzi,
                header=header,
            )
            rate.sleep()
        
   
        
