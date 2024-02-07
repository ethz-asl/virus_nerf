import numpy as np
import torch
from torch.utils.data import Dataset

from args.args import Args
from datasets.ray_utils import get_rays
from helpers.data_fcts import sensorName2ID



class DatasetBase(Dataset):
    def __init__(
            self, 
            args:Args, 
            split='train', 
        ):
        self.args = args
        self.split = split

    def __len__(self):
        return len(self.poses)
    
    def __call__(
        self, 
        batch_size:int=None,
        sampling_strategy:dict=None,
        elapse_time:float=None,
        img_idxs:torch.Tensor=None,
        pix_idxs:torch.Tensor=None,
    ):
        """
        Get some data from the dataset.
        Args:
            batch_size: number of samples; int
            sampling_strategy: dictionary containing the sampling strategy for images and pixels; dict
            elapse_time: time to elapse; float
            img_idxs: indices of images; tensor of int64 (batch_size,)
            pix_idxs: indices of pixels; tensor of int64 (batch_size,)
        Returns:
            samples: dictionary containing the sampled data; dict
        """
        # sample image and pixel indices if not provided
        if img_idxs is None or pix_idxs is None:
            img_idxs, pix_idxs = self.sampler(
                batch_size=batch_size,
                sampling_strategy=sampling_strategy,
                elapse_time=elapse_time,
            ) 

        # calculate ray origins and directions
        rays_o, rays_d = self._calcRayPoses(
            directions_dict=self.directions_dict,
            poses=self.poses,
            sensor_ids=self.sensor_ids,
            img_idxs=img_idxs,
            pix_idxs=pix_idxs,
        )

        # sample data
        ids = self.sensor_ids[img_idxs]
        rgbs = self.rgbs[img_idxs, pix_idxs, :3]
        time = self.times[img_idxs]
        samples = {
            'img_idxs': img_idxs,
            'pix_idxs': pix_idxs,
            'sensor_ids': ids.detach().clone().requires_grad_(False),
            'time': time.detach().clone().requires_grad_(False),
            'rays_o': rays_o.detach().clone().requires_grad_(True),
            'rays_d': rays_d.detach().clone().requires_grad_(True),
            'rgb': rgbs.detach().clone().requires_grad_(True),
            'depth': {},
        }
        for sensor, sensor_depths in self.depths_dict.items():
            samples['depth'][sensor] = sensor_depths[img_idxs, pix_idxs].detach().clone().requires_grad_(True)

        return samples
    
    def to(
        self, 
        device,
    ):
        """
        Move dataset to device.
        Args:
            device: device to move the dataset to; torch.device
        Returns:
            self: dataset on device; DatasetBase
        """
        self.rgbs = self.rgbs.to(device)
        self.poses = self.poses.to(device)
        self.poses_lidar = self.poses_lidar.to(device)
        self.times = self.times.to(device)
        self.sensor_ids = self.sensor_ids.to(device)
        for key in self.depths_dict.keys():
            self.depths_dict[key] = self.depths_dict[key].to(device)
        for cam_id, directions in self.directions_dict.items():
            self.directions_dict[cam_id] = directions.to(device)
        return self
    
    def getMeanHeight(
        self,
    ):
        """
        Get mean height of the images.
        Returns:
            mean_height: mean height of the images; float
        """
        mean_height = torch.mean(self.poses[:, 2, 3])
        return mean_height.item()
    
    def getSyncIdxs(
        self,
        img_idxs:torch.Tensor,
    ):
        """
        Get samples that are synchrone in time.
        Args:
            img_idxs: indices of images; tensor of int64 (batch_size,)
        Returns:
            sync_idxs: synchronized samples w.r.t. time of img_idxs; tensor of int64 (batch_size, sync_size)
        """
        time_thr = 0.1

        # determine how many samples are synchrone
        sync_size = torch.sum(torch.abs(self.times[img_idxs[0]] - self.times) < time_thr).item()

        sync_idxs = -1 * torch.ones((len(img_idxs), sync_size), dtype=torch.int32, device=self.args.device)
        for i, idx in enumerate(img_idxs):
            mask = torch.abs(self.times[idx] - self.times) < time_thr
            sync_idxs[i, :] = torch.where(mask)[0]

        # verify that all samples were updated
        if self.args.training.debug_mode:
            if torch.any(sync_idxs == -1):
                self.args.logger.error(f"DatasetBase:getSynchroneSamples: some samples were not updated correctly")
        return sync_idxs
    
    def reduceImgHeight(
        self,
        rgbs:torch.Tensor,
        directions:torch.Tensor,
        depths:np.array,
        img_wh:tuple,
        angle_min_max:tuple,
    ):
        """
        Reduce the image height to the specified range.
        Args:
            rgbs: colors; tensor of shape (N, H*W, 3)
            directions: ray directions; tensor of shape (H*W, 3)
            depths: depths; numpy array of shape (N, H*W)
            img_wh: image width and height; tuple of ints
            angle_min_max: tuple containing the min and max angles of the image to keep
        Returns:
            rgbs: colors; tensor of shape (N, H*W, 3)
            directions: ray directions; tensor of shape (H*W, 3)
            depths: depths; tensor of shape (N, H*W)
            img_wh: image width and height; tuple of ints
        """
        rgbs = rgbs.clone().detach()
        directions = directions.clone().detach()
        depths = np.copy(depths)
        W, H = img_wh
        N = rgbs.shape[0]

        # verify dimensions and reshape tensors
        if rgbs.shape[0] != depths.shape[0]:
            self.args.logger.error(f"rgbs and depths must have the same number of images")
        if rgbs.shape[1] != W*H or directions.shape[0] != W*H or depths.shape[1] != W*H:
            self.args.logger.error(f"rgbs, directions and depths must have the same number of pixels = {W*H}")
        rgbs = rgbs.reshape(N, H, W, 3)
        directions = directions.reshape(H, W, 3)
        depths = depths.reshape(N, H, W)

        # convert angles to indices
        idx_slope = H / self.args.rgbd.angle_of_view[1]
        idx_min_max = (
            max(np.floor(H/2 + idx_slope*angle_min_max[0]).astype(int), 0),
            min(np.ceil(H/2 + idx_slope*angle_min_max[1]).astype(int), H),
        )

        # reduce image height
        img_wh = (W, idx_min_max[1]-idx_min_max[0])
        rgbs = rgbs[:, idx_min_max[0]:idx_min_max[1], :, :]
        directions = directions[idx_min_max[0]:idx_min_max[1], :, :]
        depths = depths[:, idx_min_max[0]:idx_min_max[1], :]

        # reshape tensors
        rgbs = rgbs.reshape(N, img_wh[0]*img_wh[1], 3)
        directions = directions.reshape(img_wh[0]*img_wh[1], 3)
        depths = depths.reshape(N, img_wh[0]*img_wh[1])
        return rgbs, directions, depths, img_wh
    
    def _calcRayPoses(
        self,
        directions_dict:torch.Tensor,
        poses:torch.Tensor,
        sensor_ids:torch.Tensor,
        img_idxs:torch.Tensor,
        pix_idxs:torch.Tensor,
    ):
        """
        Calculate ray origins and directions for a batch of rays.
        Args:
            directions_dict: dictionary containing the directions for each sensor; dict { cam_id: directions (H*W, 3) }
            poses: poses; tensor of shape (N_dataset, 3, 4)
            sensor_ids: sensor ids; tensor of int64 (N_dataset,)
            img_idxs: indices of images; tensor of int64 (N_batch,)
            pix_idxs: indices of pixels; tensor of int64 (N_batch,)
        Returns:
            rays_o: ray origins; tensor of shape (N_batch, 3)
            rays_d: ray directions; tensor of shape (N_batch, 3)
        """
        N = img_idxs.shape[0]

        rays_o = torch.full((N, 3), np.nan, dtype=torch.float32, device=self.args.device)
        rays_d = torch.full((N, 3), np.nan, dtype=torch.float32, device=self.args.device)
        for cam_id, directions in directions_dict.items():
            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )

            idx_mask = (sensor_ids[img_idxs] == id) # (N,)
            img_idxs_temp = img_idxs[idx_mask] # (n,)
            pix_idxs_temp = pix_idxs[idx_mask] # (n,)

            rays_o_temp, rays_d_tempt = get_rays(
                directions=directions[pix_idxs_temp],
                c2w=poses[img_idxs_temp],
            ) # (n, 3), (n, 3)

            rays_o[idx_mask] = rays_o_temp # (N, 3)
            rays_d[idx_mask] = rays_d_tempt # (N, 3)

        if self.args.training.debug_mode:
            if torch.any(torch.isnan(rays_o)) or torch.any(torch.isnan(rays_d)):
                self.args.logger.error(f"DatasetBase:_calcRayPoses: some rays were not calculated correctly")

            if not torch.allclose(torch.norm(rays_d, dim=1), torch.ones((rays_d.shape[0]), device=self.args.device)):
                self.args.logger.error(f"DatasetBase::_calcRayPoses: directions are not normalized")

        return rays_o, rays_d


