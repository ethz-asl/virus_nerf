import numpy as np
import os
import pandas as pd

from rosbag_wrapper import RosbagWrapper


class PoseSync(RosbagWrapper):
    def __init__(
        self,
        data_dir:str,
        bag_name:str,
        poses_name:str,
    ) -> None:
        self.data_dir = data_dir
        self.poses_name = poses_name
        
        super().__init__(
            data_dir=data_dir,
            bag_name=bag_name,
        )
        
    def __call__(
        self,
        return_msgs:bool,
        times_dict:dict,
        masks:dict,
    ):
        """
        Sync pose with other sensors.
        Args:
            return_msgs: whether to return msgs; bool
            times_dict: dictionary with times; dict
            masks: masks of valid measurements determined by MeasSync; dict of np.array of bools
        returns:
            topics: list of topics; list of str
            msgs: list of msgs; list of list of Odometry msgs
        """
        # Sync poses
        poses_1_sync, times_1 = self._syncPose(
            stack_id=1,
            times_rs=times_dict["/CAM1/color/image_raw"],
            mask=masks["CAM1"],
        )
        poses_3_sync, times_3 = self._syncPose(
            stack_id=3,
            times_rs=times_dict["/CAM3/color/image_raw"],
            mask=masks["CAM3"],
        )
        
        # Save to CSV file
        pd.DataFrame(
            data=np.hstack((times_1[:,None], poses_1_sync)),
            columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
            dtype=np.float64,
        ).to_csv(
            path_or_buf=os.path.join(self.data_dir, 'poses', self.poses_name.replace('.csv', '_sync1.csv')),
            index=False,
        )
        pd.DataFrame(
            data=np.hstack((times_3[:,None], poses_3_sync)),
            columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
            dtype=np.float64,
        ).to_csv(
            path_or_buf=os.path.join(self.data_dir, 'poses', self.poses_name.replace('.csv', '_sync3.csv')),
            index=False,
        )
        
        if not return_msgs:
            return
        
        # Convert to msgs
        msgs_1 = self.poses2msgs(
            poses=poses_1_sync,
            times=times_1,
            frame_id="map",
        )
        msgs_3 = self.poses2msgs(
            poses=poses_3_sync,
            times=times_3,
            frame_id="map",
        )
        
        topics = ["/kiss/odometry_sync1", "/kiss/odometry_sync3"]
        msgs = [msgs_1, msgs_3]
        return topics, msgs
        
    def _syncPose(
        self,
        stack_id:int,
        times_rs:np.array=None,
        mask:np.array=None,
    ):
        """
        Synchronize poses with camera samples from one sensor stack.
        Args:
            stack_id: id of sensor stack; int
            times_rs: times of camera samples; np.array of floats (N)
            mask: mask of valid measurements determined by MeasSync; np.array of bools (N)
        Returns:
            poses_sync: synchronized poses; np.array of floats (N,7)
            times_rs: times of camera samples; np.array of floats (N)
        """
        if times_rs is None: 
            _, times_rs = self.read(
                return_time=["/CAM"+str(stack_id)+"/color/image_raw"],
            )
            
        df_poses = pd.read_csv(
            os.path.join(self.data_dir, 'poses', self.poses_name),
            dtype=np.float64,
        )
        poses = df_poses[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values
        times_poses = df_poses['time'].values
        
        idxs_below, idxs_above = self._findNeighbours(
            times_target=times_rs,
            times_source=times_poses,
        )
        
        poses_sync = self._linInterpolate(
            t1=times_poses[idxs_below],
            t2=times_poses[idxs_above],
            t=times_rs,
            x1=poses[idxs_below],
            x2=poses[idxs_above],
        )
        
        if mask is not None:
            poses_sync = poses_sync[mask]
            times_rs = times_rs[mask]
        return poses_sync, times_rs
        
    def _findNeighbours(
        self,
        times_target:np.array,
        times_source:np.array,
    ):
        """
        For each element in times_target, find the closest indices below and above of times_source.
        Args:
            times_target: times of target; np.array of floats (N)
            times_source: times of source; np.array of floats (M)
        Returns:
            idxs_below: indices of source times below target times; np.array of floats (N)
            idxs_above: indices of source times below target times; np.array of floats (N)
        """
        times_target_rep = np.tile(times_target, (len(times_source),1)).T # (N,M)
        times_source_rep = np.tile(times_source, (len(times_target),1)) # (N,M)
        
        idxs_above = np.argmax(times_source_rep >= times_target_rep, axis=1) # (N,)
        idxs_above[np.sum(times_source_rep >= times_target_rep, axis=1) == 0] = len(times_source)-1
        idxs_below = idxs_above - 1
        idxs_below[idxs_below < 0] = 0
        
        return idxs_below, idxs_above
        
    def _linInterpolate(
        self,
        t1:float,
        t2:float,
        t:float,
        x1:float,
        x2:float,
    ):
        """
        Linear interpolation between two points.
        If t<=t1 or t>=t2, then x=x1 or x=x2, respectively.
        Args:
            t1: time of first point; np.array of floats (M)
            t2: time of second point, where t2>t1; np.array of floats (M)
            t: time of interpolation; np.array of floats (N)
            x1: value of first point; np.array of floats (M,7)
            x2: value of second point; np.array of floats (M,7)
        Returns:
            x: interpolated value; np.array of floats (N,7)
        """
        if np.any(t2 < t1):
            error_mask = (t2 > t1)
            print(f"ERROR: PoseSync._linInterpolate: idxs where t2>t1: {np.where(error_mask)}, t1: {t1[error_mask]}, t2: {t2[error_mask]}")
            
        mask_smaller = (t <= t1)
        mask_larger = (t >= t2)
        mask_normal = ~ (mask_smaller | mask_larger)
        
        x = np.zeros((t.shape[0], x1.shape[1]))
        x[mask_smaller] = x1[mask_smaller]
        x[mask_larger] = x2[mask_larger]
        x[mask_normal] = ((t[mask_normal]-t1[mask_normal])/(t2[mask_normal]-t1[mask_normal]))[:,None] \
                         * (x2[mask_normal]-x1[mask_normal]) + x1[mask_normal]
        return x
    
    
def test_findNeighbours():
    pose_sync = PoseSync(
        data_dir="_",
        bag_name="_",
        poses_name="_",
    )
    
    times_target = np.array([1.5, 2.5, 3.5])
    times_source = np.array([2.0, 2.1, 3.0, 3.5])
    
    times_below, times_above = pose_sync._findNeighbours(times_target, times_source)
    

    print(f"test_findNeighbours: times_below={times_below}")
    print(f"test_findNeighbours: times_above={times_above}")

if __name__ == '__main__':
    test_findNeighbours()