import os
import numpy as np
import torch
import pandas as pd
import cv2 as cv
import matplotlib.image as mpimg

from robotathome import RobotAtHome

from datasets.ray_utils import get_ray_directions
from datasets.dataset_base import DatasetBase
from datasets.scene_rh import SceneRH
from datasets.sensor_rgbd import RGBDModel
from datasets.sensor_tof import ToFModel
from datasets.sensor_uss import USSModel
from helpers.data_fcts import sensorName2ID
from args.args import Args
from training.sampler import Sampler


class DatasetRH(DatasetBase):
    def __init__(
        self, 
        args:Args, 
        split:str='train',
        scene:SceneRH=None,
    ):

        super().__init__(args=args, split=split)

        cam_ids = [
            'RGBD_1',
            'RGBD_2',
            'RGBD_3',
            'RGBD_4',
        ]

        # load dataset
        self.rh = RobotAtHome(
            rh_path = self.args.rh.dataset_dir, 
            rgbd_path = os.path.join(self.args.rh.dataset_dir, 'files/rgbd'), 
            scene_path = os.path.join(self.args.rh.dataset_dir, 'files/scene'), 
            wspc_path = 'results', 
            db_filename = "rh.db"
        )

        # load dataframe
        self.df = self._loadRHDataframe(split=split)

        # load scene
        self.scene = scene
        if self.scene is None:
            self.scene = SceneRH(
                rh=self.rh, 
                args=self.args
            )            

        img_wh, K_dict, directions_dict = self.readIntrinsics(
            cam_ids=cam_ids,
        )
        poses, rgbs, depths_dict, sensors_dict, sensor_ids, times = self.readMeta(
            df=self.df,
            img_wh=img_wh,
            cam_ids=cam_ids,
            directions_dict=directions_dict,
        )

        self.img_wh = img_wh
        # self.K = K_dict["RGBD_1"]
        self.poses = poses
        self.rgbs = rgbs
        self.directions_dict = directions_dict
        self.sensors_dict = sensors_dict
        self.depths_dict = depths_dict
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

    def readIntrinsics(
        self,
        cam_ids,
    ):
        """
        Read camera intrinsics from the dataset.
        Args:
            cam_ids: list of camera ids, list of strings
        Returns:
            img_wh: tuple of image width and height
            K_dict: camera intrinsic matrix; dict { cam_id: tensor of shape (3, 3) }
            directions_dict: ray directions; dict { cam_id: tensor of shape (H*W, 3) }
        """
        # get image hight and width
        id = self.df["id"].to_numpy()[0]
        [rgb_f, d_f] = self.rh.get_RGBD_files(id)
        img = mpimg.imread(rgb_f)
        h, w, _ = img.shape
        img_wh = (w, h)

        # get camera intrinsics
        cx = 157.3245865
        cy = 120.0802295
        fx = 286.441384
        fy = 271.36999
        K = np.array([[fy, 0.0, cy],
                      [0.0, fx, cx], 
                      [0.0, 0.0, 1.0]])

        # get ray directions
        directions = get_ray_directions(h, w, K) # (H*W, 3) = (H, W, 3).flatten()

        # assume each camera has the same intrinsics
        directions_dict = {}
        K_dict = {}
        for cam_id in cam_ids:
            directions_dict[cam_id] = directions
            K_dict[cam_id] = K
        
        # convert to tensors
        for cam_id in cam_ids:
            directions_dict[cam_id] = torch.tensor(directions_dict[cam_id], dtype=torch.float32, requires_grad=False)
            K_dict[cam_id] = torch.tensor(K_dict[cam_id], dtype=torch.float32, requires_grad=False)

        return img_wh, K_dict, directions_dict

    def readMeta(
        self, 
        df:pd.DataFrame,
        img_wh:tuple,
        cam_ids:list,
        directions_dict:dict,
    ):
        """
        Read meta data from the dataset.
        Args:
            df: robot@home2 dataframe, pandas df
            img_wh: image width and height; tuple of ints
            cam_ids: list of camera ids, list of strings
            directions_dict: ray directions; dict { cam_id: tensor of shape (H*W, 3) }
        Returns:
            poses: tensor of shape (N_images, 3, 4) containing camera poses
            rgbs: tensor of shape (N_images, H*W, 3) containing RGB images
            depths: tensor of shape (N_images, H*W) containing depth images
            sensor_ids: id of the stack; numpy array (N_images,)
            times: timestamps in seconds starting at 0; tensor (N_images,)
        """
        df = df.copy(deep=True)

        # determine stack id: stack identity of each sample
        sensor_ids = torch.full((df.shape[0],), -1, dtype=int)
        for cam_id in cam_ids:
            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )

            sensor_id = self.rh.name2id(cam_id, "s")
            mask = np.array(df["sensor_id"] == sensor_id, dtype=bool)
            sensor_ids[mask] = id

        # read camera poses
        poses = self._readPoses(
            df=df,
        )
        poses = self._convertPoses(
            poses=poses,
        )

        # read RGBD images
        rgbs, depths = self._readImgs(
            df=df,
            img_wh=img_wh,
        )
        rgbs = self._convertColorImgs(
            rgbs=rgbs,
        )
        depths = self._convertDepthImgs(
            depths=depths,
            directions_dict=directions_dict,
            sensor_ids=sensor_ids,
        )

        # read timestamps
        times = self._readTimestamp(
            df=df,
        )
        times = self._convertTimestamps(
            times=times,
        )

        # create sensor models
        sensors_dict, depths_dict = self._createSensorModels(
            depths=depths,
            img_wh=img_wh,
            sensor_ids=sensor_ids,
        )

        return poses, rgbs, depths_dict, sensors_dict, sensor_ids, times
    
    def getIdxFromSensorName(
        self, 
        sensor_name:str,
        df:pd.DataFrame=None, 
        
    ):
        """
        Get the indices of the dataset that belong to a particular sensor.
        Args:
            sensor_name: name of the sensor, str
            df: robot@home dataframe, pandas df
        Returns:
            idxs: indices of the dataset that belong to the sensor
        """
        if df is None:
            df = self.df
        
        sensor_id = self.rh.name2id(sensor_name, "s")
        mask = np.array(df["sensor_id"] == sensor_id, dtype=bool)
        idxs = np.where(mask)[0]
        return idxs
    
    def _loadRHDataframe(
        self, 
        split:str,
    ):
        """
        Load robot@home data frame
        Args:
            split: train, val or test split, str
        Returns:
            df: rh dataframe; pandas df
        """
        # load only labeld RGBD observations
        df = self.rh.get_sensor_observations('lblrgbd') 

        # get only observations from specific home and room 
        home_id = self.rh.name2id(self.args.rh.home, "h")
        room_id = self.rh.name2id(self.args.rh.home+"_"+self.args.rh.room, "r")
        df = df[(df['home_id'] == home_id) & (df['room_id'] == room_id)]

        # split dataset
        df = self.splitDataset(
            df = df, 
            split_ratio = self.args.dataset.split_ratio, 
            split_description_path = os.path.join(self.args.rh.dataset_dir, 'files', 'rgbd', 
                                                  self.args.rh.session, self.args.rh.home, self.args.rh.room),
            split_description_name = 'split_'+self.args.rh.subsession+'.csv'
        )
        df = df[df["split"] == split]

        # keep only observations from particular sensor
        if self.args.dataset.keep_sensor != "all":
            name_idxs = self.getIdxFromSensorName(df=df, sensor_name=self.args.dataset.keep_sensor)
            df = df[name_idxs]

        # keep only first N observations
        if self.args.dataset.keep_N_observations != "all":
            df = df.iloc[:self.args.dataset.keep_N_observations,:]

        return df
    
    def _readPoses(
        self,
        df:pd.DataFrame,
    ):
        """
        Read camera poses from the dataset.
        Args:
            df: robot@home dataframe; pandas.DataFrame
        Returns:
            poses: camera poses; array of shape (N_images, 3, 4)
        """
        # get positions
        sensor_pose_x = df["sensor_pose_x"].to_numpy()
        sensor_pose_y = df["sensor_pose_y"].to_numpy()
        sensor_pose_z = df["sensor_pose_z"].to_numpy()
        p_cam2w = np.stack((sensor_pose_x, sensor_pose_y, sensor_pose_z), axis=1)

        # get orientations
        sensor_pose_yaw = df["sensor_pose_yaw"].to_numpy()
        sensor_pose_yaw -= np.deg2rad(90)
        sensor_pose_pitch = df["sensor_pose_pitch"].to_numpy()
        sensor_pose_roll = df["sensor_pose_roll"].to_numpy()

        # create rotation matrix
        R_yaw = np.stack((np.cos(sensor_pose_yaw), -np.sin(sensor_pose_yaw), np.zeros_like(sensor_pose_yaw),
                              np.sin(sensor_pose_yaw), np.cos(sensor_pose_yaw), np.zeros_like(sensor_pose_yaw),
                              np.zeros_like(sensor_pose_yaw), np.zeros_like(sensor_pose_yaw), np.ones_like(sensor_pose_yaw)), axis=1).reshape(-1, 3, 3)
        R_pitch = np.stack((np.cos(sensor_pose_pitch), np.zeros_like(sensor_pose_pitch), np.sin(sensor_pose_pitch),
                                np.zeros_like(sensor_pose_pitch), np.ones_like(sensor_pose_pitch), np.zeros_like(sensor_pose_pitch),
                                -np.sin(sensor_pose_pitch), np.zeros_like(sensor_pose_pitch), np.cos(sensor_pose_pitch)), axis=1).reshape(-1, 3, 3)
        R_roll = np.stack((np.ones_like(sensor_pose_roll), np.zeros_like(sensor_pose_roll), np.zeros_like(sensor_pose_roll),
                               np.zeros_like(sensor_pose_roll), np.cos(sensor_pose_roll), -np.sin(sensor_pose_roll),
                               np.zeros_like(sensor_pose_roll), np.sin(sensor_pose_roll), np.cos(sensor_pose_roll)), axis=1).reshape(-1, 3, 3)
        R_cam2w = np.matmul(R_yaw, np.matmul(R_pitch, R_roll))

        # create pose matrix
        poses = np.concatenate((R_cam2w, p_cam2w[:, :, np.newaxis]), axis=2) # (N_images, 3, 4)
        return poses
    
    def _readImgs(
        self, 
        df:pd.DataFrame, 
        img_wh:tuple,
    ):
        """
        Read RGBD images from the dataset.
        Args:
            df: robot@home dataframe; pandas.DataFrame
            img_wh: image width and height; tuple of ints
        Returns:
            rays: color images; array of shape (N_images, H*W, 3)
            depths: depth images; array of shape (N_images, H*W)
        """
        W, H = img_wh

        # get images
        ids = df["id"].to_numpy()
        rays = np.empty((ids.shape[0], W*H, 3))
        depths = np.empty((ids.shape[0], W*H), dtype=np.float32)
        for i, id in enumerate(ids):
            # read color and depth image
            [rgb_f, d_f] = self.rh.get_RGBD_files(id)
            rays[i,:,:] = mpimg.imread(rgb_f).reshape(W*H, 3)
            depth = cv.imread(d_f, cv.IMREAD_UNCHANGED)

            # verify depth image
            if self.args.training.debug_mode:
                if np.max(depth) > 115 or np.min(depth) < 0:
                    self.args.logger.error(f"robot_at_home.py: read_meta: depth image has invalid values")
                if not (np.allclose(depth[:,:,0], depth[:,:,1]) and np.allclose(depth[:,:,0], depth[:,:,2])):
                    self.args.logger.error(f"robot_at_home.py: read_meta: depth image has more than one channel")

            # convert depth
            depth = depth[:,:,0] # (H, W), keep only one color channel
            depths[i,:] = depth.flatten()

        return rays, depths
    
    def _readTimestamp(
        self,
        df:pd.DataFrame,
    ):
        """
        Read timestamps from the dataset.
        Args:
            df: robot@home dataframe; pandas.DataFrame
        Returns:
            timestamps: timestamps; numpy array (N_images,)
        """
        return df["timestamp"].to_numpy()
    
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
            device=self.args.device,
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
        return torch.tensor(rgbs, dtype=torch.float32, requires_grad=False, device=self.args.device)
    
    def _convertDepthImgs(
        self,
        depths:np.ndarray,
        directions_dict:np.ndarray,
        sensor_ids:np.ndarray,
    ):
        """
        Convert depth images to cube coordinates.
        Args:
            depths: depth images; array of shape (N_images, H*W)
            directions_dict: ray directions; dict { cam_id: tensor of shape (H*W, 3) }
            sensor_ids: id of the stack; int
        Returns:
            depths: depth images; tensor of shape (N_images, H*W)
        """
        # convert depth to meters
        depths = 5.0 * depths / 128.0 # (N, H*W)

        # convert depth from depth-image to depth-scan
        depths_scan = np.zeros_like(depths) # (N, H*W)
        for cam_id, directions in directions_dict.items():
            sensor_mask = (int(cam_id[-1]) == sensor_ids) # (N,)

            rs = depths / np.sqrt(1 - directions[:,0]**2 - directions[:,1]**2)[None, :] # (N, H*W)
            depths_scan[sensor_mask,:] = rs[sensor_mask,:] # (N, H*W)
        depths = depths_scan # (N, H*W)

        # set invalid depth values to nan
        depths[depths==0] = np.nan # (N, H*W)
        
        # convert depth to cube coordinate system [-0.5, 0.5]
        depths = self.scene.w2c(depths.flatten(), only_scale=True).reshape(depths.shape) # (N, H*W)
        
        # convert to tensor
        depths = torch.tensor(depths, dtype=torch.float32, requires_grad=False, device=self.args.device)
        return depths
    
    def _convertTimestamps(
        self,
        times:np.ndarray,
    ):
        """
        Convert timestamps to seconds starting at 0.
        Args:
            times: timestamps; numpy array (N_images,)
        Returns:
            times: timestamps in seconds starting at 0; tensor (N_images,)
        """
        times = times / 10000000.0
        times = times - times[0]
        return torch.tensor(times, dtype=torch.float32, requires_grad=False, device=self.args.device)
        
    def _createSensorModels(
        self, 
        depths:torch.tensor,
        img_wh:tuple,
        sensor_ids:np.ndarray,
    ):
        """
        Create sensor models for each sensor and convert depths respectively.
        Args:
            depths: depths of all images; tensor of shape (N, H*W)
            img_wh: image width and height; tuple of ints
            sensor_ids: sensor identities; numpy array (N_images,)
        Returns:
            sensors_dict: dictionary containing sensor models
            depths_dict: dictionary containing converted depths
        """
        depths = depths.detach().clone().to("cpu").numpy() # (N, H*W)

        sensors_dict = {} 
        for sensor_name in self.args.dataset.sensors:

            if sensor_name == "RGBD":
                sensors_dict["RGBD"] = RGBDModel(
                    args=self.args, 
                    img_wh=img_wh
                )
            elif sensor_name == "ToF":
                sensors_dict["ToF"] = ToFModel(
                    args=self.args, 
                    img_wh=img_wh
                )
            elif sensor_name == "USS":
                sensors_dict["USS"] = USSModel(
                    args=self.args, 
                    img_wh=img_wh,
                    sensor_ids=sensor_ids,
                )
            else:
                self.args.logger.error(f"ERROR: robot_at_home.__init__: sensor model {sensor_name} not implemented")

        depths_dict = {}
        for sensor_name in self.args.dataset.sensors:
            # convert depth with one sensor model
            depths_temp = sensors_dict[sensor_name].convertDepth(
                depths=depths,
                format="img",
            ) # (N, H*W)

            depths_dict[sensor_name] = torch.tensor(
                data=depths_temp,
                dtype=torch.float32,
                requires_grad=False,
            )

        return sensors_dict, depths_dict
    
    def splitDataset(
        self, 
        df:pd.DataFrame, 
        split_ratio:dict, 
        split_description_path:str, 
        split_description_name:str,
    ):
        """
        Split the dataset into train, val and test sets.
        Args:
            df: dataframe containing the dataset
            split_ratio: dictionary containing the split ratio for each split
            split_description_path: path to the directory containing the split description; str
            split_description_name: filename of split description; str
        Returns:
            df: dataframe containing the dataset with a new column 'split'
        """
        df = df.copy(deep=True) 

        # load split description if it exists already
        df_description = None
        if os.path.exists(os.path.join(split_description_path, 'split_description.csv')):    
            df_description = pd.read_csv(os.path.join(split_description_path, 'split_description.csv'), 
                                         index_col=0, dtype={'info':str,'train':float, 'val':float, 'test':float})
        
        # load split if it exists already
        if os.path.exists(os.path.join(split_description_path, split_description_name)):
            # split ratio must be the same as in description (last split)
            if df_description.loc[split_description_name, 'train']==split_ratio['train'] \
                and df_description.loc[split_description_name, 'val']==split_ratio['val'] \
                and df_description.loc[split_description_name, 'test']==split_ratio['test']:

                # load split and merge with df
                df_split = pd.read_csv(os.path.join(split_description_path, split_description_name))
                df = pd.merge(df, df_split, on='id', how='left')
                return df

        
        
        # get indices for each sensor
        split_idxs = {"train": np.empty(0, dtype=int), "val": np.empty(0, dtype=int), "test": np.empty(0, dtype=int)}
        for id in df["sensor_id"].unique():
            id_idxs = df.index[df["sensor_id"] == id].to_numpy()
   
            # get indices for each split
            partitions = ["train" for _ in range(int(split_ratio['train']*10))] \
                        + ["val" for _ in range(int(split_ratio['val']*10))] \
                        + ["test" for _ in range(int(split_ratio['test']*10))]
            for offset, part in enumerate(partitions):
                split_idxs[part] = np.concatenate((split_idxs[part], id_idxs[offset::10]))

        # assign split
        df.insert(1, 'split', None) # create new column for split
        df.loc[split_idxs["train"], 'split'] = 'train'
        df.loc[split_idxs["val"], 'split'] = 'val'
        df.loc[split_idxs["test"], 'split'] = 'test'

        # save split
        df_split = df[['id', 'split', 'sensor_name']].copy(deep=True)
        df_split.to_csv(os.path.join(split_description_path, split_description_name), index=False)

        # save split description
        if df_description is None:
            df_description = pd.DataFrame(columns=['info','train', 'val', 'test'])
            df_description.loc["info"] = ["This file contains the split ratios for each split file in the same directory. " \
                                          + "The Ratios must be a multiple of 0.1 and sum up to 1.0 to ensure correct splitting.", "", "", ""]
        df_description.loc[split_description_name] = ["-", split_ratio['train'], split_ratio['val'], split_ratio['test']]
        df_description.to_csv(os.path.join(split_description_path, 'split_description.csv'), index=True)

        return df
    
