import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from pcl_transformer import PCLTransformer
from pcl_loader import PCLLoader



class PCLMerger(PCLLoader):
    def __init__(
        self,
        data_dir:str,
        grid_size:float=0.05,
        occgrid_thr:int=5,
    ) -> None:
        
        super().__init__(
            data_dir=data_dir,
        )
        
        data_dir_split = data_dir.split("/")
        room = data_dir_split[-1]
        
        if room == "office_2":
            self.xyz_min = np.array([-3.0, -3.0, -1.0])
            self.xyz_max = np.array([15.0, 8.0, 5.0])
        elif room == "medium_2":
            self.xyz_min = np.array([-3.0, -3.0, -1.0])
            self.xyz_max = np.array([15.0, 9.0, 5.0])
        elif room == "corridor":
            self.xyz_min = np.array([0.0, -3.0, -1.0])
            self.xyz_max = np.array([40.0, 3.0, 5.0])
        elif room == "office_scan":
            self.xyz_min = np.array([-4.0, -2.0, -1.0])
            self.xyz_max = np.array([5.0, 8.0, 5.0])
        elif room == "medium_scan_2":
            self.xyz_min = np.array([-5.0, -2.0, -1.0])
            self.xyz_max = np.array([15.0, 7.0, 5.0])
        else:
            print(f"ERROR: pcl_merger.py:__init__ room={room} not supported")
            sys.exit()
            
        self.grid_size = grid_size
        self.occgrid_thr = occgrid_thr
        
    def merge(
        self,
        times_filename:str,
        poses_filename:str,
        pcl_dir:str,
        output_filename:str,
        plot_stats:bool=False,
    ):
        _, times = self.loadPoses(
            pose_format='vector',
            filename=times_filename,
        )
        poses, poses_times = self.loadPoses(
            pose_format='vector',
            filename=poses_filename,
        )
        lidar_filenames = self.getFiles(
            pcl_dir=pcl_dir,
        )
        lidar_times = self._filename2time(
            filenames=lidar_filenames,
        )
        
        # match times
        idxs = self.matchTimes(
            times_subset=times,
            times=lidar_times,
        )
        lidar_filenames = np.array(lidar_filenames)[idxs]
        idxs = self.matchTimes(
            times_subset=times,
            times=poses_times,
        )
        poses = poses[idxs]
        
        # merge the point clouds
        idxs_max = self._pos2idx(
            xyz=self.xyz_max.reshape((1, 3))
        ).flatten()
        occgrid = np.zeros((idxs_max[0]+1, idxs_max[1]+1, idxs_max[2]+1), dtype=np.uint16)
        for i in range(len(lidar_filenames)):
            # transform pointcloud to world frame
            trans = PCLTransformer(
                q=poses[i, 3:],
                t=poses[i, :3],
            )
            xyz = self.loadPCL(
                filename=os.path.join(pcl_dir, lidar_filenames[i]),
            )
            xyz = trans.transformPointcloud(
                xyz=xyz,
            )
            
            # add pointcloud to occupancy grid
            idxs = self._pos2idx(
                xyz=xyz,
            )
            occgrid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] += 1
            
            # check overflow
            if (occgrid >= 65535).any():
                print(f"WARNING: pcl_merger.py:merge occupancy grid overflow possible at next iteration")
                
        if plot_stats:
            plt.hist(occgrid.flatten(), bins=np.linspace(0, 50, 51))
            plt.vlines(self.occgrid_thr-1, 0, 1e6, colors='r', linestyles='dashed', label=f"threshold={self.occgrid_thr}")
            plt.title(f"Histogram of occupancy grid: removing {np.sum((occgrid < self.occgrid_thr) & (occgrid >= 1)) / np.sum(occgrid >= 1):.3}% of points")
            plt.xlabel("Number of points in cell")
            plt.ylabel("Number of elements")
            plt.yscale("log") 
            plt.xlim([0, 50])
            plt.legend()
            plt.show()
            
            
        # threshold occupancy grid and convert to pointcloud
        occgrid[occgrid < self.occgrid_thr] = 0
        idxs = np.argwhere(occgrid > 0)
        xyz = self._idx2pos(
            idxs=idxs,
        )
        
        # save pointcloud
        self.savePCL(
            filename=output_filename,
            xyz=xyz,
        )
    
    def matchTimes(
        self,
        times_subset:np.ndarray,
        times:np.ndarray,
        threshold:float=0.001,
    ):
        """
        Match times between times and subset of times.
        Args:
            times_subset: subset of times; array (N,)
            times: times; array (M,)
            threshold: maximum time difference; float
        Returns:
            idxs: indices of matched times; array (N,)
        """
        if times_subset.shape[0] > times.shape[0]:
            print(f"ERROR: pcl_merger.py:matchTimes times_subset.shape[0] > times.shape[0]")
            sys.exit()
        
        t1, t2 = np.meshgrid(times_subset, times, indexing='ij') # (N, M), (N, M)
        mask = (np.abs(t1 - t2) < threshold) # (N, M)
        
        if not (np.sum(mask, axis=1) == 1).all():
            print(f"ERROR: pcl_merger.py:matchTimes np.sum(mask, axis=0) != 1")
            sys.exit()
        
        idxs = np.argmax(mask, axis=1) # (N,)
        return idxs
    
    def difference(
        self,
        pcl1_filename:str,
        pcl2_filename:str,
        output1_filename:str,
        output2_filename:str,
    ):
        xyz1 = self.loadPCL(
            filename=pcl1_filename,
        )
        xyz2 = self.loadPCL(
            filename=pcl2_filename,
        )
        
        idxs_max = self._pos2idx(
            xyz=self.xyz_max.reshape((1, 3))
        ).flatten()
        occgrid1 = np.zeros((idxs_max[0]+1, idxs_max[1]+1, idxs_max[2]+1), dtype=np.uint16)
        occgrid2 = np.zeros((idxs_max[0]+1, idxs_max[1]+1, idxs_max[2]+1), dtype=np.uint16)
        
        idxs1 = self._pos2idx(
            xyz=xyz1,
        )
        occgrid1[idxs1[:, 0], idxs1[:, 1], idxs1[:, 2]] = 1
        idxs2 = self._pos2idx(
            xyz=xyz2,
        )
        occgrid2[idxs2[:, 0], idxs2[:, 1], idxs2[:, 2]] = 1
        
        occgrid_diff12 = occgrid1 - occgrid2
        occgrid_diff21 = occgrid2 - occgrid1
        
        idxs_diff12 = np.argwhere(occgrid_diff12 > 0)
        idxs_diff21 = np.argwhere(occgrid_diff21 > 0)
        
        xyz_diff12 = self._idx2pos(
            idxs=idxs_diff12,
        )
        xyz_diff21 = self._idx2pos(
            idxs=idxs_diff21,
        )
        
        self.savePCL(
            filename=output1_filename,
            xyz=xyz_diff12,
        )
        self.savePCL(
            filename=output2_filename,
            xyz=xyz_diff21,
        )
            
    def _pos2idx(
        self,
        xyz:np.ndarray,
    ):
        """
        Convert position to index.
        Args:
            xyz: position; array (N, 3)
        Returns:
            idx: index; array (N, 3)
        """
        xyz = xyz[(xyz <= self.xyz_max).all(axis=1) & (xyz >= self.xyz_min).all(axis=1)]
        idxs = (xyz - self.xyz_min) / self.grid_size
        return np.round(idxs).astype(np.uint32)
    
    def _idx2pos(
        self,
        idxs:np.ndarray,
    ):
        """
        Convert index to position.
        Args:
            idx: index; array (N, 3)
        Returns:
            xyz: position; array (N, 3)
        """
        xyz = idxs * self.grid_size + self.grid_size/2 + self.xyz_min
        return xyz
    
    
def run_PCLMerger():
    data_dir = "/home/spadmin/catkin_ws_ngp/data/medium_scan_2"
    
    if not os.path.exists(os.path.join(data_dir, "maps")):
        os.makedirs(os.path.join(data_dir, "maps"))
    
    pcl_merger = PCLMerger(
        data_dir=data_dir,
        grid_size=0.03,
        occgrid_thr=2,
    )
    pcl_merger.merge(
        times_filename='poses/poses_lidar_balm.csv',
        poses_filename='poses/poses_lidar_balm.csv',
        pcl_dir='lidars/filtered',
        output_filename='maps/map_balm.pcd',
        plot_stats=True,
    )
    pcl_merger.merge(
        times_filename='poses/poses_lidar_balm.csv',
        poses_filename='poses/poses_lidar.csv',
        pcl_dir='lidars/filtered',
        output_filename='maps/map_kiss.pcd',
        plot_stats=True,
    )
    
    # calculate difference
    pcl_merger.difference(
        pcl1_filename='maps/map_balm.pcd',
        pcl2_filename='maps/map_kiss.pcd',
        output1_filename='maps/map_balm-kiss.pcd',
        output2_filename='maps/map_kiss-balm.pcd',
    )
    

    
if __name__ == "__main__":
    run_PCLMerger()
    # comparePoses()
    







# def comparePoses():
#     data_dir = "/home/spadmin/catkin_ws_ngp/data/corridor"
#     pcl_merger = PCLMerger(
#         data_dir=data_dir,
#     )
    
#     poses_lidar, times_lidar = pcl_merger.loadPoses(
#         pose_format='vector',
#         filename='poses/poses_lidar.csv',
#     )
#     poses_lidar_balm, times_lidar_balm = pcl_merger.loadPoses(
#         pose_format='vector',
#         filename='poses/poses_lidar_balm.csv',
#     )
#     idxs = pcl_merger.matchTimes(
#         times_subset=times_lidar_balm,
#         times=times_lidar,
#     )
#     poses_lidar = poses_lidar[idxs]
    
#     if not np.allclose(poses_lidar, poses_lidar_balm):
#         print(f"ERROR: pcl_merger.py:comparePoses poses_lidar != poses_lidar_balm")
#         xyz_error = np.linalg.norm(poses_lidar[:, :3] - poses_lidar_balm[:, :3], axis=1)
#         print(f"xyz_error: max={np.max(xyz_error)}, min={np.min(xyz_error)}, mean={np.mean(xyz_error)}")
    
#     # poses_lidar, times_lidar = pcl_merger.loadPoses(
#     #     pose_format='matrix_times',
#     #     filename='balm/alidarPose.csv',
#     # )
#     # poses_lidar_balm = pcl_merger.loadPoses(
#     #     pose_format='matrix',
#     #     filename='balm/poses_lidar_balm.csv',
#     # )

#     # if not np.allclose(poses_lidar, poses_lidar_balm):
#     #     print(f"ERROR: pcl_merger.py:comparePoses poses_lidar != poses_lidar_balm")
#     #     xyz_error = np.linalg.norm(poses_lidar[:, :3, 3] - poses_lidar_balm[:, :3, 3], axis=1)
#     #     print(f"xyz_error: max={np.max(xyz_error)}, min={np.min(xyz_error)}, mean={np.mean(xyz_error)}")
    
# def evaluate(
#         self,
#         map_filename:str,
#         poses_filename:str,
#         depth_dir:str,
#         eval_N_samples:int=100,
#         eval_N_points_per_sample:int=100,
#     ):
#         # load map, depth images and poses
#         xyz_map = self.loadPCL(
#             filename=map_filename,
#         )
#         depth_dir = os.path.join(self.data_dir, depth_dir)
#         depth_files_temp = [f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f))]
#         depth_files = [f for f in depth_files_temp if f.endswith(".npy")]
#         poses, _ = self.loadPoses(
#             pose_format='vector',
#             filename=poses_filename,
#         )
        
#         # subsample depth images
#         depth_files = np.random.choice(depth_files, size=eval_N_samples, replace=False)
        
#         pcl_creator = PCLCreatorRS(
#             data_dir=self.data_dir,
#             sensor_id="CAM1",
#         )
        
#         num_pts_depth = 0
#         num_pts_
#         for i, f in enumerate(depth_files):
#             # transform depth image to pointcloud
#             depth = np.load(os.path.join(depth_dir, f))
#             xyz = pcl_creator.depth2pcl(
#                 depth=depth,
#             )
            
#             # subsample pointcloud
#             idxs = np.random.choice(xyz.shape[0], size=eval_N_points_per_sample, replace=False)
#             xyz = xyz[idxs]
            
#             # convert pointcloud to world frame
#             trans = PCLTransformer(
#                 q=poses[i, 3:],
#                 t=poses[i, :3],
#             )
#             xyz = trans.transformPointcloud(
#                 xyz=xyz,
#             )