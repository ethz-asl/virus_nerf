import numpy as np
import pandas as pd
import os
import shutil


class LidarSync():
    def __init__(
        self,
        data_dir,
    ) -> None:
        
        self.data_dir = data_dir
        
        
        self.times = self._loadTimes(
            data_dir=data_dir,
        )
        
    def sync(
        self,
    ):
        """
        Sync lidar data
        """
        times = self._loadTimes(
            data_dir=self.data_dir,
        )
        raw_data_dir = os.path.join(self.data_dir, 'lidars/raw')
        sync_data_dir = os.path.join(self.data_dir, 'lidars/sync')
        
        lidar_files = os.listdir(raw_data_dir)
        lidar_times = np.array([float(lidar_file[:-4]) for lidar_file in lidar_files], dtype=np.float64)
        
        # np.set_printoptions(precision=20)
        # print(f"files: {lidar_files[:5]}")
        # print(f"times: {lidar_times[:5]}")
        
        for time in times:
            closest_time, closest_idx = self._findClosestTime(
                time=time,
                times=lidar_times,
            )
            
            if abs(closest_time - time) > 0.1:
                print(f'Closest time to {time} is {closest_time} which is more than 0.1 seconds away')
            
            shutil.copy(
                src=os.path.join(raw_data_dir, lidar_files[closest_idx]),
                dst=os.path.join(sync_data_dir, f'{time}.pcd'),
            )
            
    def _loadTimes(
        self,
        data_dir,
    ):
        """
        Load synchronized times from poses. Use camera 1 as reference.
        Args:
            data_dir: path to data directory
        Returns:
            times: synchronized times
        """
        times_path = os.path.join(data_dir, 'poses/poses_sync1_cam_robot.csv')
        df = pd.read_csv(times_path)
        times = df['time'].to_numpy(dtype=np.float64)
        return times
    
    def _findClosestTime(
        self,
        time,
        times,
    ):
        """
        Find closest time in times to time
        Args:
            time: time to find closest time to; float
            times: list of times; np.array of floats
        Returns:
            closest_time: closest time to time
            closest_idx: index of closest time
        """
        closest_idx = np.argmin(np.abs(times - time))
        closest_time = times[closest_idx]
        return closest_time, closest_idx
    
    
    
def main_syncLidar():
    
    data_dir = "/home/spadmin/catkin_ws_ngp/data/office_2"
    
    lidar_sync = LidarSync(
        data_dir=data_dir,
    )
    lidar_sync.sync()
    
if __name__ == '__main__':
    main_syncLidar()