import numpy as np 
import pandas as pd
from pypcd4 import PointCloud
import os
import sys



class PCLLoader():
    def __init__(
        self,
        data_dir:str,
    ) -> None:
        self.data_dir = data_dir
        
    def getFiles(
        self,
        pcl_dir:str,
    ):
        """
        Get all files in the maps directory.
        Args:
            pcl_dir: directory of the point clouds; str
        Returns:
            filenames: list of files in the maps directory; list of str (N,)
        """
        maps_dir = os.path.join(self.data_dir, pcl_dir)
        return [f for f in os.listdir(maps_dir) if os.path.isfile(os.path.join(maps_dir, f))]
    
    def getTimes(
        self,
        pcl_dir:str,
    ):
        """
        Get all times of the point clouds.
        Args:
            pcl_dir: directory of the point clouds; str
        Returns:
            times: times of the point clouds; array of floats
            filenames: filenames of the point clouds; list of str
        """
        filenames = self.getFiles(
            pcl_dir=pcl_dir,
        )
        times = self._filename2time(
            filenames=filenames,
        )
        return times, filenames
        
    def loadLatestPCL(
        self,
    ):
        """
        Load the latest point cloud in time from the maps directory.
        Returns:
            xyz: point cloud; np.array of shape (N, 3)
        """
        # load all time stamps of the point clouds assuming that the filenames are the time stamps
        times, filenames = self.getTimes()
        
        # determine the latest point cloud
        idx_max = np.argmax(times)
        
        return self.loadPCL(
            filename=filenames[idx_max]
        )
    
    def loadPCL(
        self,
        filename:str,
    ):
        """
        Load a point cloud from the maps directory.
        Args:
            filename: filename of the point cloud; str
        Returns:
            xyz: point cloud; np.array of shape (N, 3)
        """
        pc = PointCloud.from_path(os.path.join(self.data_dir, filename))
        xyz = pc.numpy(
            fields=['x', 'y', 'z'],
        )
        return xyz
    
    def savePCL(
        self,
        filename:str,
        xyz:np.ndarray,
    ):
        """
        Save a point cloud to the maps directory.
        Args:
            filename: filename of the point cloud; str
            xyz: point cloud; np.array of shape (N, 3)
        """
        fields = ("x", "y", "z")
        types = (np.float32, np.float32, np.float32)
        pc = PointCloud.from_points(xyz.astype(np.float32), fields, types)
        
        pc.save(os.path.join(self.data_dir, filename))
        
    def loadPoses(
        self,
        pose_format:str,
        filename:str,
    ):
        """
        Load lidar poses from csv file estimated by BALM.
        Args:
            pose_format: format of the poses; str
                    'vector': [x, y, z, qx, qy, qz, qw]
                    'matrix': transformation matrix (4, 4)
            filename: filename of the poses; str
        Returns:
            poses: lidar poses; array (N, 7)
            times: lidar times; array (N,)
        """
        if pose_format == 'vector':
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(self.data_dir, filename),
                dtype=np.float64,
            )
            poses = df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy(dtype=np.float64)
            times = df['time'].to_numpy(dtype=np.float64)
            return poses, times
        elif pose_format == 'matrix':
            poses = np.loadtxt(
                fname=os.path.join(self.data_dir, filename),
                delimiter=",",
                dtype=np.float64,
            )
            poses = poses.reshape((-1, 4, 4))
            return poses
        elif pose_format == 'matrix_times':
            poses = np.loadtxt(
                fname=os.path.join(self.data_dir, filename),
                delimiter=",",
                dtype=np.float64,
            )
            poses = poses.reshape((-1, 4, 4))
            times = np.copy(poses[:, 3, 3])
            poses[:, 3, 3] = 1.0
            return poses, times
        else:
            print(f"ERROR: pcl_merger.py:_loadPoses format={format} not supported")
            sys.exit()
            
    def savePoses(
        self,
        filename:str,
        pose_format:str,
        poses:np.ndarray,
        times:np.ndarray=None,
    ):
        """
        Save lidar poses to csv file.
        Args:
            filename: filename of the poses; str
            pose_format: format of the poses; str
                         'vector': [x, y, z, qx, qy, qz, qw]
                         'matrix': transformation matrix (4, 4)
            poses: lidar poses; array (N, 7) or (N, 4, 4)
            times: lidar times; array (N,)
        """
        if pose_format == 'vector':
            poses_time = np.hstack((times.reshape(-1, 1), poses))
            df = pd.DataFrame(
                data=poses_time,
                columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
            )
            df.to_csv(
                path_or_buf=os.path.join(self.data_dir, filename),
                index=False,
            )
        elif pose_format == 'matrix':
            poses = poses.reshape((-1, 4))
            np.savetxt(
                fname=os.path.join(self.data_dir, filename),
                X=poses,
                delimiter=",",
                fmt="%.9f",
            )
        elif pose_format == 'matrix_times':
            poses[:, 3, 3] = times
            poses = poses.reshape((-1, 4))
            np.savetxt(
                fname=os.path.join(self.data_dir, filename),
                X=poses,
                delimiter=",",
                fmt="%.9f",
            )
        else:
            print(f"ERROR: pcl_merger.py:_savePoses pose_format={pose_format} not supported")
            sys.exit()
    
    def renamePCL(
        self,
        prefix:str,
        pcl_dir:str,
    ):
        filenames = self.getFiles(
            pcl_dir=pcl_dir,
        )
        times = self._filename2time(
            filenames=filenames
        )
        idxs = np.argsort(times)
        filenames = [filenames[idx] for idx in idxs]
        
        for i, f in enumerate(filenames):
            os.rename(os.path.join(self.data_dir, pcl_dir, f), os.path.join(self.data_dir, pcl_dir, prefix + str(i) + ".pcd"))
    
    def _filename2time(
        self,
        filenames:list,
    ):
        """
        Convert filenames to time stamps.
        Args:
            filenames: filename of the point cloud; list of str (N,)
        Returns:
            times: time of the point cloud; array (N,)
        """
        times = [float(t[:-4]) for t in filenames]
        return np.array(times)
        

    
    
    
def test_pcl_loader():
    data_dir = "/home/spadmin/catkin_ws_ngp/data/test"
    
    pcl_loader = PCLLoader(
        data_dir=data_dir,
        pcl_dir="maps",
    )
    xyz = pcl_loader.loadLatestPCL()
    
    print(xyz.shape)
    print(xyz[:10])    
    
    pcl_loader.renamePCL(
        prefix="full",
        pcl_dir="lidar/raw",
    )
    
def test_rename():
    data_dir = "/home/spadmin/catkin_ws_ngp/data/test"
    
    pcl_loader = PCLLoader(
        data_dir=data_dir,
        pcl_dir="lidar_scans",
    )
    pcl_loader.renamePCL(
        prefix="full",
        pcl_dir="lidar/raw",
    )
    
if __name__ == "__main__":
    # test_pcl_loader()
    test_rename()