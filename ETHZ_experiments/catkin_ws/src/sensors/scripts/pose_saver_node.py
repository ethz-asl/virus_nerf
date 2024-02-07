import rospy
from nav_msgs.msg import Odometry
import pandas as pd
import os
import numpy as np

from pcl_tools.pcl_transformer import PCLTransformer


class PoseSaver():
    def __init__(
        self,
        topic_pose:str,
        data_dir:str,
        output_format:str,
    ):
        self.poses = []
        self.times = []
        self.pcl_counter = 0
        self.data_dir = data_dir
        self.output_format = output_format
        
        poses_dir = os.path.join(self.data_dir, "poses")
        if not os.path.exists(poses_dir):
            rospy.loginfo(f'Creating directory: {poses_dir}')
            os.makedirs(poses_dir)
        
        # ROS
        rospy.init_node('PCLSaver_node')
        self.sub_pose = None
        self.topic_pose = topic_pose
    
    def subscribe(
        self,
    ):
        self.sub_pose = rospy.Subscriber(self.topic_pose, Odometry, self.callbackPose)
        
        rospy.on_shutdown(self.savePoses)

        rospy.spin()
        
    def savePoses(
        self
    ):
        # convert pose to numpy array
        if self.output_format == 'matrix':
            Ts = np.zeros((len(self.poses), 4, 4))
            for i, pose in enumerate(self.poses):
                trans = PCLTransformer(
                    t=pose[:3],
                    q=pose[3:7],
                )
                T = trans.getTransform(
                    type="matrix",
                )
                Ts[i] = T
            Ts = Ts.reshape((-1,4))
        elif self.output_format == 'quaternion':
            Ts = np.array(self.poses)
        else:
            rospy.logerr(f'Invalid output format: {self.output_format}')
            
        # convert time to numpy array
        times = np.array(self.times)
            
        # Save pose ant time to CSV file
        file_path = os.path.join(self.data_dir, "poses", 'poses_lidar.csv')
        if self.output_format == 'matrix':
            np.savetxt(file_path.replace('.csv', '_matrix.csv'), Ts, delimiter=',')
            np.savetxt(file_path.replace('.csv', '_time.csv'), times, delimiter=',')
        elif self.output_format == 'quaternion':
            pd.DataFrame(
                data=np.hstack((times.reshape((-1,1)), Ts)),
                columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
                dtype=np.float64,
            ).to_csv(file_path, index=False)
        else:
            rospy.logerr(f'Invalid output format: {self.output_format}')

    def callbackPose(
        self, 
        msg:Odometry,
    ):
        p = msg.pose.pose
        
        self.poses.append([
            p.position.x, 
            p.position.y, 
            p.position.z,
            p.orientation.x, 
            p.orientation.y, 
            p.orientation.z, 
            p.orientation.w,
        ])
        self.times.append(msg.header.stamp.to_sec())


def main():
    pcl_saver = PoseSaver(
        topic_pose=rospy.get_param('topic_pose'),
        data_dir=rospy.get_param('data_dir'),
        output_format=rospy.get_param('output_format'),
    )
    pcl_saver.subscribe()

if __name__ == "__main__":
    main()
    
    