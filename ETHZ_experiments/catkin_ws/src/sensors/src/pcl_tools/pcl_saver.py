import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from nav_msgs.msg import Odometry
import open3d as o3d
import os
import shutil
import numpy as np

from pcl_tools.pcl_transformer import PCLTransformer



class PCLSaver():
    def __init__(
        self,
        topic_pcl:str,
        topic_pose:str,
        save_dir:str,
    ):
        self.poses = []
        self.pcl_counter = 0
        self.save_dir = save_dir
        
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # ROS
        rospy.init_node('PCLSaver_node')
        self.sub_pcl = None
        self.sub_pose = None
        self.topic_pcl = topic_pcl
        self.topic_pose = topic_pose
    
    def subscribe(
        self,
    ):
        self.sub_pcl = rospy.Subscriber(self.topic_pcl, PointCloud2, self.callbackPCL)
        self.sub_pose = rospy.Subscriber(self.topic_pose, Odometry, self.callbackPose)
        
        rospy.on_shutdown(self.savePoses)

        rospy.spin()
        
    def savePoses(
        self
    ):
        Ts = np.zeros((len(self.poses), 4, 4))
        for i, pose in enumerate(self.poses):
            T = pose.getTransform(
                type="matrix",
            )
            Ts[i] = T
        Ts = Ts.reshape((-1,4))
            
        # Save to CSV file
        file_path = os.path.join(self.save_dir, 'alidarPose.csv')
        np.savetxt(file_path, Ts, delimiter=',')

    def callbackPCL(
        self, 
        msg:PointCloud2,
    ):
        # Convert ROS PointCloud2 message to PCL
        cloud = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # pcl_list = [ [p[0],p[1],p[2]] for p in cloud ]
        xyz = np.array(list(cloud), dtype=np.float32)
        
        # Create Open3D PointCloud
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(xyz)

        # Save to .pcd
        file_path = os.path.join(self.save_dir, f'full{self.pcl_counter}.pcd')
        o3d.io.write_point_cloud(file_path, cloud_o3d)
        
        self.pcl_counter += 1

    def callbackPose(
        self, 
        msg:Odometry,
    ):
        p = msg.pose.pose
        pose = PCLTransformer(
            t=[p.position.x, p.position.y, p.position.z],
            q=[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
        )
        self.poses.append(pose)