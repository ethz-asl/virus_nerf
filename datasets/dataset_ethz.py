import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2 as cv
from scipy.spatial.transform import Rotation

from datasets.ray_utils import get_ray_directions
from datasets.dataset_base import DatasetBase
from datasets.scene_ethz import SceneETHZ
from datasets.splitter_ethz import SplitterETHZ
from datasets.sensor_rgbd import RGBDModel
from datasets.sensor_tof import ToFModel
from datasets.sensor_uss import USSModel
from args.args import Args
from training.sampler import Sampler
from helpers.data_fcts import sensorName2ID, sensorID2Name
from ETHZ_experiments.catkin_ws.src.sensors.src.pcl_tools.pcl_loader import PCLLoader
from ETHZ_experiments.catkin_ws.src.sensors.src.pcl_tools.pcl_transformer import PCLTransformer
from ETHZ_experiments.catkin_ws.src.sensors.src.pcl_tools.pcl_creator import PCLCreatorUSS, PCLCreatorToF


class DatasetETHZ(DatasetBase):
    def __init__(
        self, 
        args:Args, 
        split:str='train',
        scene:SceneETHZ=None,
    ):
        self.time_start = None
        

        super().__init__(
            args=args, 
            split=split
        )

        dataset_dir = self.args.ethz.dataset_dir
        data_dir = os.path.join(dataset_dir, self.args.ethz.room)

        # load scene
        self.scene = scene
        if scene is None:
            self.scene = SceneETHZ(
                args=self.args,
                data_dir=data_dir,
            )

        # split dataset
        splitter = SplitterETHZ(
            args=args,
        )
        split_masks = splitter.splitDataset(
            split=split,
        )

        # load camera intrinsics
        img_wh, K_dict, directions_dict = self.readIntrinsics(
            dataset_dir=dataset_dir,
            data_dir=data_dir,
            cam_ids=self.args.ethz.cam_ids,
        )

        # load samples
        poses, poses_lidar, rgbs, depths_dict, sensors_dict, sensor_ids, times = self.readMetas(
            data_dir=data_dir,
            cam_ids=self.args.ethz.cam_ids,
            img_wh=img_wh,
            split_masks=split_masks,
            directions_dict=directions_dict,
        )

        self.img_wh = img_wh
        self.poses = poses
        self.poses_lidar = poses_lidar
        self.directions_dict = directions_dict
        self.rgbs = rgbs
        self.depths_dict = depths_dict
        self.sensors_dict = sensors_dict
        self.sensor_ids = sensor_ids
        self.times = times

        # initialize sampler
        self.sampler = Sampler(
            args=args,
            dataset_len=len(self),
            img_wh=self.img_wh,
            sensors_dict=self.sensors_dict,
            times=self.times,
        )

    def getIdxFromSensorName(
        self, 
        sensor_name:str,
        
    ):
        """
        Get the indices of the dataset that belong to a particular sensor.
        Args:
            sensor_name: name of the sensor, str
        Returns:
            idxs: indices of the dataset that belong to the sensor
        """
        stack_id = self.sensor_ids.detach().clone().cpu().numpy()
        id = sensorName2ID(
            sensor_name=sensor_name,
            dataset=self.args.dataset.name,
        )

        mask = (stack_id == id)
        idxs = np.where(mask)[0]
        return idxs
    
    def getSensorNameFromIdx(
        self,
        idxs:np.ndarray,
    ):
        """
        Get the sensor name from the indices of the dataset.
        Args:
            idxs: indices of the dataset that belong to the sensor
        Returns:
            sensor_name: name of the sensor, str
        """
        sensor_id = self.sensor_ids[idxs].detach().clone().cpu().numpy()
        sensor_name = sensorID2Name(
            sensor_id=sensor_id,
            dataset=self.args.dataset.name,
        )
        return sensor_name

    
    def getLidarMaps(
        self,
        img_idxs:np.ndarray,
    ):
        """
        Load LiDAR maps and convert them into world coordinate system.
        Args:
            img_idxs: indices of samples; numpy array of shape (N,)
        Returns:
            xyzs: list of point clouds; list of length N of numpy arrays of shape (M, 3)
            poses: poses in world coordinates; list of numpy arrays of shape (N, 3, 4)
        """
        # get times and poses of samples in world coordinate system
        times = self.times[img_idxs].clone().detach().cpu().numpy() # (N,)
        poses = self.poses_lidar[img_idxs].clone().detach().cpu().numpy() # (N, 3, 4)
        poses[:,:,3]  = self.scene.c2w(pos=poses[:,:,3], copy=False) # (N, 3)
        
        # load lidar file names and times
        pcl_loader = PCLLoader(
            data_dir=os.path.join(self.args.ethz.dataset_dir, self.args.ethz.room),
        )
        lidar_times, lidar_files = pcl_loader.getTimes(
            pcl_dir='lidars/filtered',
        )
        sort_idxs = np.argsort(lidar_times)
        lidar_files = np.array(lidar_files)[sort_idxs]
        lidar_times = lidar_times[sort_idxs]
        lidar_times = self.normalizeTimes(
            times=lidar_times,
        )

        # find corresponding lidar file to each sample
        m1, m2 = np.meshgrid(times, lidar_times, indexing='ij')
        mask = (np.abs(m1-m2) < 0.05)
        lidar_idxs = np.argmax(mask, axis=1)
        lidar_files = lidar_files[lidar_idxs]
        if self.args.training.debug_mode:
            if not np.all(np.sum(mask, axis=1) == np.ones((mask.shape[0]))):
                self.args.logger.error(f"DatasetETHZ::getLidarMaps: multiple or no lidar files found for one sample")
                self.args.logger.error(f"time: {times}")
                self.args.logger.error(f"lidar_times: {lidar_times[np.where(mask)[1]]}")
        
        xyzs = []
        for i, f in enumerate(lidar_files):
            # load point cloud
            xyz = pcl_loader.loadPCL(
                filename=os.path.join('lidars/filtered', f),
            ) # (M, 3)

            # convert robot coordinate system to world coordinate system
            trans = PCLTransformer(
                t=poses[i,:3,3],
                R=poses[i,:3,:3],
            )
            xyz = trans.transformPointcloud(
                xyz=xyz,
            )
            xyzs.append(xyz)
            
        return xyzs, poses
    
    def getRobotPose2D(
        self,
        img_idxs:np.ndarray,
        pose_in_world_coords:bool,
    ):
        """
        Get the 2D robot pose.
        Args:
            img_idxs: indices of samples; numpy array of shape (N,)
            pose_in_world_coords: if True, return pose in world coordinates; bool
        Returns:
            pos_cam1: robot position; numpy array of shape (N, 2)
            pos
            angle: robot angle; numpy array of shape (N,)
        """
        W, H = self.img_wh
        N = img_idxs.shape[0]

        # get lidar poses
        poses_lidar = self.poses_lidar[img_idxs].detach().clone().cpu().numpy() # (N*2, 3, 4)
        rays_o_lidar = poses_lidar[:, :3, 3] # (N*2, 3)
        rot = Rotation.from_matrix(poses_lidar[:, :3, :3]) # (N*2,)
        angles_lidar = rot.as_euler('zyx', degrees=False)[:,0] # (N*2,)

        # get camera poses
        sync_idxs = self.getSyncIdxs(
            img_idxs=img_idxs,
        ) # (N, 2)
        sync_idxs = sync_idxs.reshape(-1) # (N*2,)

        rays_o, rays_d = self._calcRayPoses(
            directions_dict=self.directions_dict,
            poses=self.poses,
            sensor_ids=self.sensor_ids,
            img_idxs=sync_idxs,
            pix_idxs=torch.tensor(N*2 * [0.5*W*(H+1)], dtype=torch.int32, device=self.args.device)
        ) # (N*2, 3)
        rays_o = rays_o.detach().cpu().numpy() # (N*2, 3)
        rays_d = rays_d.detach().cpu().numpy() # (N*2, 3)
        angles = np.arctan2(rays_d[:,1], rays_d[:,0]) # (N*2,)

        sensor_ids = self.sensor_ids[sync_idxs].detach().clone().cpu().numpy() # (N*2,)
        pos = {
            "LiDAR": rays_o_lidar[:,:2], # (N, 2)
            "CAM1": rays_o[sensor_ids==1,:2], # (N, 2)
            "CAM3": rays_o[sensor_ids==3,:2], # (N, 2)
        }
        orientation = {
            "LiDAR": angles_lidar, # (N,)
            "CAM1": angles[sensor_ids==1], # (N,)
            "CAM3": angles[sensor_ids==3], # (N,)
        }

        if pose_in_world_coords:
            for k in pos.keys():
                pos[k] = self.scene.c2w(pos=pos[k], copy=False) # (N, 2)

        if self.args.training.debug_mode:
            for p, o in zip(pos.values(), orientation.values()):
                if (o.shape[0] != N) or (p.shape[0] != N):
                    self.args.logger.error(f"DatasetETHZ::getRobotPose2D: mask should size of N")
                    sys.exit()

        return pos, orientation

    def getFieldOfView(
        self,
        img_idxs:np.ndarray,
    ):
        """
        Get the field of view of a sensor.
        Args:
            img_idxs: indices of samples; numpy array of shape (N,)
        Returns:
            fov: field of view of the sensors; dict of { sensor: { camera: array of shape (N,) } }
            pos: robot position; dict of { camera: array of shape (N, 2) }
            orientation: robot orientation; dict of { camera: array of shape (N,) }
        """
        # get 2D pose of lidar and cameras
        pos, orientation = self.getRobotPose2D(
            img_idxs=img_idxs,
            pose_in_world_coords=True,
        )

        # get field of view
        fov_tof = np.deg2rad([-self.args.tof.angle_of_view[0]/2, self.args.tof.angle_of_view[0]/2]) # (2,)
        fov_uss = np.deg2rad([-self.args.uss.angle_of_view[0]/2, self.args.uss.angle_of_view[0]/2]) # (2,)
        fov_lidar = np.deg2rad(self.args.lidar.angle_min_max[self.args.ethz.room]) # (2,)

        fov = {
            "USS": {
                "CAM1": orientation["CAM1"][:,None] + fov_uss, # (N,2)
                "CAM3": orientation["CAM3"][:,None] + fov_uss, # (N,2)
            },
            "ToF": {
                "CAM1": orientation["CAM1"][:,None] + fov_tof, # (N,2)
                "CAM3": orientation["CAM3"][:,None] + fov_tof, # (N,2)
            },
            "LiDAR": {
                "LiDAR": orientation["LiDAR"][:,None] + fov_lidar, # (N,2)
            },
            "NeRF": { 
                "LiDAR": np.ones((img_idxs.shape[0], 2)) * np.deg2rad([-180, 180]), # (N,2)
            },
        }

        # normalize angles
        for sensor in fov.keys():
            for camera in fov[sensor].keys():
                fov[sensor][camera][fov[sensor][camera] > np.pi] -= 2*np.pi
                fov[sensor][camera][fov[sensor][camera] < -np.pi] += 2*np.pi

        return fov, pos, orientation

    def readIntrinsics(
        self,
        dataset_dir:str,
        data_dir:str,
        cam_ids:list,
    ):
        """
        Read camera intrinsics from the dataset.
        Args:
            dataset_dir: path to dataset directory; str
            data_dir: path to data directory; str
            cam_ids: list of camera ids; list of str
        Returns:
            img_wh: tuple of image width and height
            K_dict: camera intrinsic matrix dictionary; dict oftensor of shape (3, 3)
            directions_dict: ray directions dictionary; dict of tensor of shape (H*W, 3)
        """
        # get image width and height
        img_path = os.path.join(data_dir, 'measurements/CAM1_color_image_raw', 'img0.png')
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        h, w, _ = img.shape
        img_wh = (w, h)

        # get camera intrinsics
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(dataset_dir, 'camera_intrinsics.CSV'),
            dtype={'cam_id': str, 'fx': np.float64, 'fy': np.float64, 'cx': np.float64, 'cy': np.float64},
        )
        K_dict = {}
        for cam_id in cam_ids:
            df_cam = df[df["cam_id"]==cam_id]
            K_dict[cam_id] = np.array([[df_cam['fx'].values[0], 0.0, df_cam['cx'].values[0]],
                                        [0.0, df_cam['fy'].values[0], df_cam['cy'].values[0]], 
                                        [0.0, 0.0, 1.0]]) # (3, 3)

        # get ray directions and normalize them
        directions_dict = {}
        for cam_id in cam_ids:
            directions = get_ray_directions(h, w, K_dict[cam_id]) # (H*W, 3)
            directions_dict[cam_id] = directions / np.linalg.norm(directions, axis=1, keepdims=True) # (H*W, 3)

            if self.args.training.debug_mode:
                if not torch.allclose(torch.norm(directions_dict[cam_id], dim=1), torch.ones((directions_dict[cam_id].shape[0]))):
                    self.args.logger.error(f"DatasetETHZ::readIntrinsics: directions are not normalized")

        # convert numpy arrays to tensors
        for cam_id in cam_ids:
            K_dict[cam_id] = torch.tensor(K_dict[cam_id], dtype=torch.float32, requires_grad=False)
            directions_dict[cam_id] = directions_dict[cam_id].to(dtype=torch.float32)
            directions_dict[cam_id].requires_grad = False

        return img_wh, K_dict, directions_dict

    def readMetas(
        self,
        data_dir:str,
        cam_ids:list,
        img_wh:tuple,
        split_masks:dict,
        directions_dict:dict,
    ):
        """
        Read all samples from the dataset.
        Args:
            data_dir: path to data directory; str
            cam_ids: list of camera ids; list of str
            img_wh: image width and height; tuple of ints
            split_mask: mask of splits; dict of { sensor type: bool array of shape (N_all_splits,) }
            directions_dict: ray directions; dict of { sensor type: array of shape (N_images, H*W, 3) }
        Returns:
            poses: camera poses; array of shape (N_images, 3, 4)
            poses_lidar: lidar poses; array of shape (N_images, 3, 4)
            rgbs: ray origins; array of shape (N_images, H*W, 3)
            depths_dict: dictionary of depth samples; dict of { sensor type: array of shape (N_images, H*W) }
            sensors_dict: dictionary of sensor models; dict of { sensor: sensor model }
            sensor_ids: stack identity number of sample; tensor of shape (N_images,)
            times: time of sample in seconds starting at 0; tensor of shape (N_images,)
        """
        # pose data
        poses, poses_lidar, sensor_ids, times = self._readPoses(
            data_dir=data_dir,
            cam_ids=cam_ids,
            split_masks=split_masks,
        ) # (N, 3, 4),  (N, 3, 4), (N,), (N,)
        poses = self._convertPoses(
            poses=poses,
        ) # (N, 3, 4)
        poses_lidar = self._convertPoses(
            poses=poses_lidar,
        ) # (N, 3, 4)

        # image color data
        rgbs, rgbs_sensor_ids = self._readColorImgs(
            data_dir=data_dir,
            cam_ids=cam_ids,
            img_wh=img_wh,
            split_masks=split_masks,
        ) # (N, H*W, 3), (N,)
        if self.args.training.debug_mode:
            if not np.all(sensor_ids == rgbs_sensor_ids):
                self.args.logger.error(f"DatasetETHZ::read_meta: stack ids do not match")
        rgbs = self._convertColorImgs(
            rgbs=rgbs,
        )

        # depth data
        depths_dict = {}
        sensors_dict = {}

        if "RGBD" in self.args.dataset.sensors:
            depths, sensor_ids = self._readDepthImgs(
                data_dir=data_dir,
                cam_ids=cam_ids,
                img_wh=img_wh,
                split_masks=split_masks,
            )
            if self.args.training.debug_mode and not np.all(sensor_ids == rgbs_sensor_ids):
                self.args.logger.error(f"DatasetETHZ::read_meta: stack ids do not match")

            rs_depths, rs_sensor_model = self._convertDepthImgs(
                depths=depths,
                directions_dict=directions_dict,
                sensor_ids=sensor_ids,
                img_wh=img_wh,
            )
            depths_dict["RGBD"] = rs_depths
            sensors_dict["RGBD"] = rs_sensor_model
        
        if "USS" in self.args.dataset.sensors:
            uss_meass, uss_sensor_ids, times = self._readUSS(
                data_dir=data_dir,
                cam_ids=cam_ids,
                split_masks=split_masks,
            ) # (N,), (N,)
            if self.args.training.debug_mode:
                if not np.all(sensor_ids == uss_sensor_ids):
                    self.args.logger.error(f"DatasetETHZ::read_meta: uss_sensor_ids ids do not match")
                if not np.allclose(times, times):
                    self.args.logger.error(f"DatasetETHZ::read_meta: times do not match")

            uss_depths, uss_sensors_model = self._convertUSS(
                meass=uss_meass,
                sensor_ids=uss_sensor_ids,
                img_wh=img_wh,
            ) # (N, H*W), dict { cam_id : USSModel }
            depths_dict["USS"] = uss_depths
            sensors_dict["USS"] = uss_sensors_model

        if "ToF" in self.args.dataset.sensors:
            tof_meass, tof_meas_stds, tof_sensor_ids, times = self._readToF(
                data_dir=data_dir,
                cam_ids=cam_ids,
                split_masks=split_masks,
            ) # (N, 64), (N, 64), (N,)
            if self.args.training.debug_mode:
                if not np.all(sensor_ids == tof_sensor_ids):
                    self.args.logger.error(f"DatasetETHZ::read_meta: tof_sensor_ids ids do not match")
                if not np.allclose(times, times):
                    self.args.logger.error(f"DatasetETHZ::read_meta: times do not match")
            
            tof_depths, tof_stds, tof_sensors_model = self._convertToF(
                meass=tof_meass,
                meas_stds=tof_meas_stds,
                img_wh=img_wh,
            ) # (N, H*W), (N, H*W), dict { cam_id : ToFModel }
            depths_dict["ToF"] = tof_depths
            sensors_dict["ToF"] = tof_sensors_model

        # convert stack ids and times to tensor
        sensor_ids = torch.tensor(sensor_ids, dtype=torch.uint8, requires_grad=False)
        times = torch.tensor(times, dtype=torch.float64, requires_grad=False)

        return poses, poses_lidar, rgbs, depths_dict, sensors_dict, sensor_ids, times
    
    def _readPoses(
        self,
        data_dir:str,
        cam_ids:list,
        split_masks:dict,
    ):
        """
        Read poses from the dataset for each camera.
        Args:
            cam_ids: list of camera ids; list of str
            data_dir: path to data directory; str
            split_masks: mask of split; bool array of shape (N_all_splits,)
        Returns:
            poses: camera poses; array of shape (N, 3, 4)
            poses_lidar: lidar poses; array of shape (N, 3, 4)
            sensor_ids: stack identity number of sample; array of shape (N,)
            times: time of sample in seconds starting at 0; array of shape (N,)
        """
        poses = np.zeros((0, 3, 4))
        poses_lidar = np.zeros((0, 3, 4))
        sensor_ids = np.zeros((0))
        times = np.zeros((0))
        for cam_id in cam_ids:
            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )

            if self.args.ethz.use_optimized_poses:
                poses_name = 'poses_cam_balm_sync' + str(id) + '.csv'
                poses_lidar_name = 'poses_lidar_balm_sync' + str(id) + '.csv'
            else:
                poses_name = 'poses_cam_sync' + str(id) + '.csv'
                poses_lidar_name = 'poses_lidar_sync' + str(id) + '.csv'

            df = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, 'poses', poses_name),
                dtype=np.float64,
            )
            df_lidar = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, 'poses', poses_lidar_name),
                dtype=np.float64,
            )

            time = df["time"].to_numpy()
            time = time[split_masks[cam_id]]

            time_lidar = df_lidar["time"].to_numpy()
            time_lidar = time_lidar[split_masks[cam_id]]

            # verify time
            if self.args.training.debug_mode:
                if not np.allclose(time, time_lidar, atol=1e-6):
                    self.args.logger.error(f"DatasetETHZ::_readPoses: time_lidar is not consistent")
                    print(f"time: {time}")
                    print(f"time_lidar: {time_lidar}")

            pose = np.zeros((np.sum(split_masks[cam_id]), 3, 4))
            for i, pose_i in enumerate(np.arange(df.shape[0])[split_masks[cam_id]]):
                trans = PCLTransformer(
                    t=[df["x"][pose_i], df["y"][pose_i], df["z"][pose_i]],
                    q=[df["qx"][pose_i], df["qy"][pose_i], df["qz"][pose_i], df["qw"][pose_i]],
                )
                T = trans.getTransform(
                    type="matrix",
                ) # (4, 4)
                pose[i] = T[:3,:] # (3, 4)

            pose_lidar = np.zeros((np.sum(split_masks[cam_id]), 3, 4))
            for i, pose_i in enumerate(np.arange(df_lidar.shape[0])[split_masks[cam_id]]):
                trans = PCLTransformer(
                    t=[df_lidar["x"][pose_i], df_lidar["y"][pose_i], df_lidar["z"][pose_i]],
                    q=[df_lidar["qx"][pose_i], df_lidar["qy"][pose_i], df_lidar["qz"][pose_i], df_lidar["qw"][pose_i]],
                )
                T = trans.getTransform(
                    type="matrix",
                ) # (4, 4)
                pose_lidar[i] = T[:3,:] # (3, 4)

            poses = np.concatenate((poses, pose), axis=0) # (N, 3, 4)
            poses_lidar = np.concatenate((poses_lidar, pose_lidar), axis=0) # (N, 3, 4)
            sensor_ids = np.concatenate((sensor_ids, np.ones((pose.shape[0]))*int(cam_id[-1])), axis=0) # (N,)
            times = np.concatenate((times, time), axis=0) # (N,)

        times = self.normalizeTimes(
            times=times,
        )

        return poses, poses_lidar, sensor_ids, times
    
    def _readColorImgs(
        self,
        data_dir:str,
        cam_ids:list,
        img_wh:tuple,
        split_masks:dict,
    ):
        """
        Read color images from the dataset for each camera.
        Args:
            data_dir: path to data directory; str
            cam_ids: list of camera ids; list of str
            img_wh: image width and height; tuple of ints
            split_mask: mask of splits; dict of { sensor type: bool array of shape (N_all_splits,) }
        Returns:
            rgbs_dict: color images; array of shape (N, H*W, 3)
            sensor_ids: stack identity number of sample; array of shape (N,)
        """
        W, H = img_wh

        rgbs = np.zeros((0, H*W, 3))
        sensor_ids = np.zeros((0))
        for cam_id in cam_ids:
            rgb_path = os.path.join(data_dir, 'measurements/'+cam_id+'_color_image_raw') 
            rgb_files = np.array(['img'+str(i)+'.png' for i in range(split_masks[cam_id].shape[0])])
            rgb_files = rgb_files[split_masks[cam_id]]

            rgbs_temp = np.zeros((len(rgb_files), H*W, 3))
            for i, f in enumerate(rgb_files):
                rgb_file = os.path.join(rgb_path, f)
                rgb = cv.imread(rgb_file, cv.IMREAD_COLOR) # (H, W, 3)
                rgbs_temp[i] = rgb.reshape(H*W, 3) # (H*W, 3)

            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )

            rgbs = np.concatenate((rgbs, rgbs_temp), axis=0) # (N, H*W, 3)
            sensor_ids = np.concatenate((sensor_ids, id*np.ones((rgbs_temp.shape[0]))), axis=0) # (N,)

        return rgbs, sensor_ids

    def _readDepthImgs(
        self,
        data_dir:str,
        cam_ids:list,
        img_wh:tuple,
        split_masks:dict,
    ):
        """
        Read depth images from the dataset for each camera.
        Args:
            cam_ids: list of camera ids; list of str
            data_dir: path to data directory; str
            img_wh: image width and height; tuple of ints
            split_masks: mask of splits; dict of { sensor type: bool array }
        Returns:
            depths: depth images; array of shape (N, H*W)
            sensor_ids: stack identity number of sample; array of shape (N,)
        """
        W, H = img_wh

        depths = np.zeros((0, H*W))
        sensor_ids = np.zeros((0))
        for cam_id in cam_ids:
            depth_path = os.path.join(data_dir, 'measurements/'+cam_id+'_aligned_depth_to_color_image_raw')
            depth_files = np.array(['img'+str(i)+'.npy' for i in range(split_masks[cam_id].shape[0])])
            depth_files = depth_files[split_masks[cam_id]]

            depths_temp = np.zeros((len(depth_files), H*W))
            for i, f in enumerate(depth_files):

                depth = np.load(
                    file=os.path.join(depth_path, f),
                )
                depths_temp[i] = depth.flatten() # (H*W)

            depths = np.concatenate((depths, depths_temp), axis=0) # (N, H*W)
            sensor_ids = np.concatenate((sensor_ids, np.ones((depths_temp.shape[0]))*int(cam_id[-1])), axis=0) # (N,)

        return depths, sensor_ids
    
    def _readUSS(
        self,
        data_dir:str,
        cam_ids:list,
        split_masks:dict,
    ):
        """
        Read USS measurements from the dataset for each camera.
        Args:
            cam_ids: list of camera ids; list of str
            data_dir: path to data directory; str
            split_massk: mask of splits; dict of { sensor type: bool array }
        Returns:
            meass: USS measurements; array of shape (N_images,)
            sensor_ids: stack identity number of sample; array of shape (N_images,)
        """
        meass = np.zeros((0))
        sensor_ids = np.zeros((0))
        times = np.zeros((0))
        for cam_id in cam_ids:
            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, 'measurements/USS'+str(id)+'.csv'),
                dtype=np.float64,
            )
            meass_temp = df["meas"].to_numpy()
            meass_temp = meass_temp[split_masks[cam_id]]

            time = df["time"].to_numpy()
            time = time[split_masks[cam_id]]

            meass = np.concatenate((meass, meass_temp), axis=0) # (N,)
            sensor_ids = np.concatenate((sensor_ids, np.ones((meass_temp.shape[0]))*int(cam_id[-1])), axis=0) # (N,)
            times = np.concatenate((times, time), axis=0) # (N,)

        times = self.normalizeTimes(
            times=times,
        )

        return meass, sensor_ids, times
    
    def _readToF(
        self,
        data_dir:str,
        cam_ids:list,
        split_masks:dict,
    ):
        """
        Read Tof measurements from the dataset for each camera.
        Args:
            data_dir: path to data directory; str
            cam_ids: list of camera ids; list of str
            split_masks: mask of splits; dict of { sensor type: bool array }
        Returns:
            meass: USS measurements; array of shape (N_images, 64)
            meas_stds: USS measurements; array of shape (N_images, 64)
            sensor_ids: stack ids; array of shape (N_images,)
            times: time of sample in seconds starting at 0; array of shape (N,)
        """
        meass = np.zeros((0, 64))
        meas_stds = np.zeros((0, 64))
        sensor_ids = np.zeros((0))
        times = np.zeros((0))
        for cam_id in cam_ids:
            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, 'measurements/TOF'+str(id)+'.csv'),
                dtype=np.float64,
            )

            time = df["time"].to_numpy()
            time = time[split_masks[cam_id]]

            meass_temp = np.zeros((df.shape[0], 64))
            stds = np.zeros((df.shape[0], 64))
            for i in range(64):
                meass_temp[:,i] = df["meas_"+str(i)].to_numpy()
                stds[:,i] = df["stds_"+str(i)].to_numpy()
            
            meass_temp = meass_temp[split_masks[cam_id]]
            stds = stds[split_masks[cam_id]]

            meass = np.concatenate((meass, meass_temp), axis=0) # (N, 64)
            meas_stds = np.concatenate((meas_stds, stds), axis=0)
            sensor_ids = np.concatenate((sensor_ids, np.ones((meass_temp.shape[0]))*int(cam_id[-1])), axis=0) # (N,)
            times = np.concatenate((times, time), axis=0) # (N,)
            
        times = self.normalizeTimes(
            times=times,
        )

        return meass, meas_stds, sensor_ids, times
    
    def _convertPoses(
        self,
        poses:dict,
    ):
        """
        Convert poses to cube coordinates.
        Args:
            poses: camera poses; array of shape (N_images, 3, 4)
        Returns:
            poses: camera poses in cube coordinates; array of shape (N_images, 3, 4)
        """
        # convert positions from world to cube coordinate system
        xyz = poses[:,:,3] # (N, 3)
        xyz = self.scene.w2c(pos=xyz, copy=False) # (N, 3)
        poses[:,:,3] = xyz # (N, 3, 4)
        
        # convert array to tensor
        poses = torch.tensor(
            data=poses,
            dtype=torch.float32,
            requires_grad=False,
        )
        return poses
    
    def _convertColorImgs(
        self,
        rgbs:np.ndarray,
    ):
        """
        Convert color images to tensors.
        Args:
            rgbs: color images; array of shape (N_images, H*W, 3)
        Returns:
            rgbs: color images; tensor of shape (N_images, H*W, 3)
        """
        rgbs /= 255.0 # (N, H*W, 3)
        rgbs = torch.tensor(rgbs, dtype=torch.float32, requires_grad=False)
        return rgbs

    def _convertDepthImgs(
        self,
        depths:np.ndarray,
        directions_dict:np.ndarray,
        sensor_ids:np.ndarray,
        img_wh:tuple,
    ):
        """
        Convert depth images to cube coordinates.
        Args:
            depths: depth images; array of shape (N_images, H*W)
            directions_dict: ray directions; dict { cam_id: tensor of shape (H*W, 3) }
            sensor_ids: id of the stack; int
            img_wh: image width and height; tuple of ints
        Returns:
            depths: depth images in cube coordinates; array of shape (N_images, H*W)
            sensors_model: RGBD sensor models; RGBDModel
        """
        # convert depth to meters
        depths = 0.001 * depths # (N, H*W)

        # convert depth from depth-image to depth-scan
        depths_scan = np.zeros_like(depths) # (N, H*W)
        for cam_id, directions in directions_dict.items():
            directions = directions.detach().clone().cpu().numpy() # (H*W, 3)

            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )

            sensor_mask = (id == sensor_ids) # (N,)

            depths_temp = depths / directions[:,2].reshape(1,-1) # (N, H*W)
            depths_scan[sensor_mask,:] = depths_temp[sensor_mask,:] # (N, H*W)
        depths = depths_scan # (N, H*W)

        # set invalid depth values to nan
        depths[depths==0.0] = np.nan # (N, H*W)
        
        # convert depth to cube coordinate system [-0.5, 0.5]
        depths = self.scene.w2c(depths.flatten(), only_scale=True).reshape(depths.shape) # (N, H*W)
        
        # convert to tensor
        depths = torch.tensor(depths, dtype=torch.float32, requires_grad=False)

        # create sensor model 
        sensors_model = RGBDModel(
            args=self.args, 
            img_wh=img_wh,
        )
        return depths, sensors_model
    
    def _convertUSS(
        self,
        meass:dict,
        sensor_ids:np.array,
        img_wh:tuple,
    ):
        """
        Convert USS measurement to depth in cube coordinates.
        Args:
            meass: dictionary containing USS measurements; array of shape (N_images,)
            sensor_ids: stack identity number of sample; array of shape (N_images,)
            img_wh: image width and height; tuple of ints
        Returns:
            depths: converted depths; array of shape (N_images, H*W)
            sensors_model: USS sensor model; USSModel
        """
        pcl_creator = PCLCreatorUSS(
            W=1,
            H=1,
        )
        
        # convert USS measurements to depth in meters
        depths_sensor = np.zeros((meass.shape[0])) # (N)
        for i, meas in enumerate(meass):
            depths_sensor[i] = pcl_creator.meas2depth(
                meas=meas,
            )

        # convert depth in meters to cube coordinates [-0.5, 0.5]
        depths_sensor = self.scene.w2c(depths_sensor.flatten(), only_scale=True) # (N,)

        # create sensor model
        sensors_model = USSModel(
            args=self.args, 
            img_wh=img_wh,
            sensor_ids=sensor_ids,
        )

        # convert depth
        depths_sensor = sensors_model.convertDepth(
            depths=depths_sensor,
            format="sensor",
        ) # (N, H*W)

        # convert depth to tensor
        depths =  torch.tensor(
            data=depths_sensor,
            dtype=torch.float32,
            requires_grad=False,
        )
        return depths, sensors_model

    def _convertToF(
        self,
        meass:np.array,
        meas_stds:np.array,
        img_wh:tuple,
    ):
        """
        Convert ToF measurement to depth in cube coordinates.
        Args:
            meass: dictionary containing ToF measurements; array of shape (N_images, 64,)
            meas_stds: dictionary containing ToF measurement standard deviations; array of shape (N_images, 64,)
            img_wh: image width and height; tuple of ints
        Returns:
            depths: converted depths; array of shape (N_images, H*W)
            stds: converted standard deviations; array of shape (N_images, H*W)
            sensor_model: ToF sensor model; ToFModel
        """
        pcl_creator = PCLCreatorToF(
            W=8,
            H=8,
        )
        
        # convert ToF measurements to depth in meters
        depths_sensor = np.zeros((meass.shape[0], 8, 8)) # (N, 8, 8)
        stds_sensor = np.zeros((meas_stds.shape[0], 8, 8)) # (N, 8, 8)
        for i in range(meass.shape[0]):
            depths_sensor[i] = pcl_creator.meas2depth(
                meas=meass[i],
            ) # (8, 8)
            stds_sensor[i] = pcl_creator.meas2depth(
                meas=meas_stds[i],
            ) # (8, 8)

        # convert depth in meters to cube coordinates [-0.5, 0.5]
        depths_sensor = self.scene.w2c(depths_sensor.flatten(), only_scale=True).reshape(-1, 64) # (N, 8*8)
        stds_sensor = self.scene.w2c(stds_sensor.flatten(), only_scale=True).reshape(-1, 64) # (N, 8*8)

        # create sensor model
        sensor_model = ToFModel(
            args=self.args, 
            img_wh=img_wh,
        )

        # mask pixels that are outside of the field of view
        depths_img = sensor_model.convertDepth(
            depths=depths_sensor,
            format="sensor",
        ) # (N, H*W)
        stds_img = sensor_model.convertDepth(
            depths=stds_sensor,
            format="sensor",
        ) # (N, H*W)

        # convert depth to tensor
        depths =  torch.tensor(
            data=depths_img,
            dtype=torch.float32,
            requires_grad=False,
        )
        stds =  torch.tensor(
            data=stds_img,
            dtype=torch.float32,
            requires_grad=False,
            )
        return depths, stds, sensor_model
    
    def normalizeTimes(
        self,
        times:np.ndarray,
    ):
        """
        Normalize times that it starts with 0.
        Args:
            times: time of sample in seconds starting at 0; array of shape (N,)
        Returns:
            times: normalized time; array of shape (N,)
        """
        if self.time_start is None:
            self.time_start = np.min(times)

        times -= self.time_start
        return times
    

    

    

   