import numpy as np
import torch

from args.args import Args
from datasets.sensor_base import SensorModel
from helpers.data_fcts import sensorID2Name


class USSModel(SensorModel):
    def __init__(
        self, 
        args:Args,
        img_wh:tuple,
        sensor_ids:np.ndarray,
    ) -> None:
        """
        USS sensor model.
        Args:
            args: arguments; Args
            img_wh: image width and height, tuple of int
            sensor_ids: sensor identities for every image; array of ints (N,)
        """
        SensorModel.__init__(
            self, 
            args=args, 
            img_wh=img_wh,
        )
        
        self.mask = self._createMask() # (H*W,)

        self.num_imgs = sensor_ids.shape[0]

        self.imgs_stats = {}
        for id in np.unique(sensor_ids):
            name = sensorID2Name(
                sensor_id=id, 
                sensor_type="USS",
                dataset=self.args.dataset.name,
            )

            img_idxs = np.where(sensor_ids == id)[0]

            self.imgs_stats[name] = {
                "id": id,
                "img_idxs": torch.tensor(img_idxs, dtype=torch.int32).to(self.args.device),
                "pix_idxs": torch.randint(0, self.W*self.H, size=(img_idxs.shape[0],), device=self.args.device, dtype=torch.int32),
                "depths": np.inf * torch.ones((img_idxs.shape[0]), dtype=torch.float32).to(self.args.device),
                "counts": torch.zeros((img_idxs.shape[0]), dtype=torch.int32).to(self.args.device),
            }

    def getStatsForBatch( 
        self,
        batch_img_idxs:torch.Tensor,
    ):
        """
        Get statistics for batch.
        Args:
            batch_img_idxs: image indices; tensor of shape (N_batch,)
        Returns:
            batch_pix_idxs: pixel indices; tensor of shape (N_batch,)
            batch_depths: minimum depth per batch; tensor of shape (N_batch,)
            batch_counts: update counts per batch; tensor of shape (N_batch,)
        """
        # get stats for all images
        imgs_pix_idxs = -1 * torch.ones((self.num_imgs), dtype=torch.int32).to(self.args.device) # (N_imgs,)
        imgs_depths = -1 * torch.ones((self.num_imgs), dtype=torch.float32).to(self.args.device) # (N_imgs,)
        imgs_counts = -1 * torch.ones((self.num_imgs), dtype=torch.int32).to(self.args.device) # (N_imgs,)
        for stats in self.imgs_stats.values():
            imgs_pix_idxs[stats["img_idxs"]] = stats["pix_idxs"] # (N_imgs,)
            imgs_depths[stats["img_idxs"]] = stats["depths"] # (N_imgs,)
            imgs_counts[stats["img_idxs"]] = stats["counts"] # (N_imgs,)

        # check if all minimum depths have been updated
        if self.args.training.debug_mode:
            if torch.any(imgs_depths < 0):
                self.args.logger.error(f"USSModel.updateDepthMin: imgs_depths < 0")
        
        # convert minimum pixels, depths and counts to batch
        batch_pix_idxs = imgs_pix_idxs[batch_img_idxs].clone().detach() # (N_batch,)
        batch_depths = imgs_depths[batch_img_idxs].clone().detach() # (N_batch,)
        batch_counts = imgs_counts[batch_img_idxs].clone().detach() # (N_batch,)
        return batch_pix_idxs, batch_depths, batch_counts

    def convertDepth(
        self, 
        depths:np.array,
        format:str="img",
    ):
        """
        Convert depth img using ToF sensor model. Set all unknown depths to nan.
        Args:
            depths: depth img
            format: depths format; str
                    "img": depth per camera pixel; depths array of shape (N, H*W)
                    "sensor": depth per ToF pixel; depths array of shape (N,)
        Returns:
            depths_out: depth img converted to ToF sensor array; array of shape (N, H*W)
        """
        depths = np.copy(depths) # (N, H*W) or (N,)
        depths_out = np.full((depths.shape[0], self.W*self.H), np.nan) # (N, H*W)
        fov_mask = self.mask.detach().clone().cpu().numpy() # (H*W,)

        if format == "img":
            d_min = np.nanmin(depths[:, fov_mask], axis=1) # (N,)
        elif format == "sensor":
            d_min = depths # (N,)
        else:
            self.args.logger.error(f"Unknown depth format: {format}")

        depths_out[:, fov_mask] = d_min[:,None] # (N, H*W)
        return depths_out   

    def updateStats( 
        self,
        depths:torch.Tensor,
        data:dict,
    ):
        """
        Update the minimum depth of each image and the corresponding pixel index.
        Args:
            depths: depth of forward pass; tensor of shape (N_batch,)
            data: data; dict
                    'img_idxs': image indices; tensor of shape (N_batch,)
                    'pix_idxs': pixel indices; tensor of shape (N_batch,)
                    'sensor_ids': sensor ids; tensor of shape (N_batch,)
        Returns:
            batch_min_depths: minimum depth per batch; tensor of shape (N_batch,)
            batch_min_counts: updated counts per batch; tensor of shape (N_batch,)
        """
        for stats in self.imgs_stats.values():
            stats = self._updateSensorStats(
                stats=stats,
                batch_depths=depths,
                data=data,
            )

        batch_min_pix_idxs, batch_min_depths, batch_min_counts = self.getStatsForBatch(
            batch_img_idxs=data["img_idxs"],
        )
        return batch_min_depths, batch_min_counts

    def _updateSensorStats(
        self,
        stats:dict,
        batch_depths:torch.Tensor,
        data:dict,
    ):
        """
        Update the minimum depth of each image and the corresponding pixel index.
        Args:
            stats: sensor stats; dict
                    'id': sensor id; int
                    'img_idxs': image indices; tensor of shape (N_sensor,)
                    'pix_idxs': pixel indices; tensor of shape (N_sensor,)
                    'depths': minimum depth; tensor of shape (N_sensor,)
                    'counts': update counts; tensor of shape (N_sensor,)
            batch_depths: depth of forward pass; tensor of shape (N_batch,)
            data: data; dict
                    'img_idxs': image indices; tensor of shape (N_batch,)
                    'pix_idxs': pixel indices; tensor of shape (N_batch,)
                    'sensor_ids': sensor ids; tensor of shape (N_batch,)
        Returns:
            stats: updated sensor stats; dict
        """
        sensor_id = stats["id"]
        sensor_min_img_idxs = stats["img_idxs"] # (N_sensor,)
        sensor_min_pix_idxs = stats["pix_idxs"] # (N_sensor,)
        sensor_min_depths = stats["depths"] # (N_sensor,)
        sensor_min_counts = stats["counts"] # (N_sensor,)

        batch_img_idxs = data["img_idxs"] # (N_batch,)
        batch_pix_idxs = data["pix_idxs"] # (N_batch,)
        batch_ids = data["sensor_ids"] # (N_batch,)

        # use only samples in field-of-view and of particular sensor to update stats
        fov_mask = torch.tensor(self.mask[batch_pix_idxs], dtype=torch.bool).to(self.args.device) # (N_batch,)
        sensor_mask = (batch_ids == sensor_id) # (N_batch,)
        mask = fov_mask & sensor_mask # (N_batch,)

        batch_img_idxs = batch_img_idxs[mask] # (n_batch,)
        batch_pix_idxs = batch_pix_idxs[mask] # (n_batch,)
        batch_depths = batch_depths[mask] # (n_batch,)

        # deterimne minimum depth contained in batch for every image
        batch_min_depths = np.inf * torch.ones((self.num_imgs, len(batch_img_idxs)), dtype=torch.float).to(self.args.device) # (N_imgs, n_batch)
        batch_min_depths[batch_img_idxs, np.arange(len(batch_img_idxs))] = batch_depths # (N_imgs, n_batch)
        batch_min_depths, min_idxs = torch.min(batch_min_depths, dim=1) # (N_imgs,), (N_imgs,)
        batch_min_pix_idxs = batch_pix_idxs[min_idxs] # (N_imgs,)

        # deterimne minimum depth contained in batch for particular sensor
        batch_min_depths = batch_min_depths[sensor_min_img_idxs] # (N_sensor,)
        batch_min_pix_idxs = batch_min_pix_idxs[sensor_min_img_idxs] # (N_sensor,)

        # update minimum depth and minimum pixel indices
        sensor_min_depths = torch.where(
            condition=(batch_min_pix_idxs == sensor_min_pix_idxs),
            input=batch_min_depths,
            other=torch.minimum(batch_min_depths, sensor_min_depths)
        ) # (N_sensor,)
        sensor_min_pix_idxs = torch.where(
            condition=(batch_min_depths <= sensor_min_depths),
            input=batch_min_pix_idxs,
            other=sensor_min_pix_idxs,
        ) # (N_sensor,)

        # update minimum counts
        batch_counts = torch.zeros((self.num_imgs), dtype=torch.int32).to(self.args.device) # (N_imgs,)
        batch_counts[batch_img_idxs] = 1 # (N_imgs,)
        batch_counts = batch_counts[sensor_min_img_idxs] # (N_sensor,)
        sensor_min_counts = sensor_min_counts + batch_counts # (N_sensor,)

        # update stats
        stats["img_idxs"] = sensor_min_img_idxs.to(dtype=torch.int32)
        stats["pix_idxs"] = sensor_min_pix_idxs.to(dtype=torch.int32)
        stats["depths"] = sensor_min_depths.to(dtype=torch.float32)
        stats["counts"] = sensor_min_counts.to(dtype=torch.int32)
        return stats

    def _createMask(
        self,
    ) -> torch.Tensor:
        """
        Create mask for ToF sensor.
        Returns:
            mask: mask for ToF sensor; tensor of shape (H*W,)
        """
        # define USS opening angle
        pix_wh = self.AoV2pixel(
            aov_sensor=self.args.uss.angle_of_view
        ) # (2,)
        pix_wh = (pix_wh/2.0).astype(np.int32) # convert diameter to radius

        # create mask
        m1, m2 = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        m1 = m1 - self.H/2 
        m2 = m2 - self.W/2
        mask = (m1/pix_wh[1])**2 + (m2/pix_wh[0])**2 < 1 # (H, W), ellipse
        mask = torch.tensor(mask.flatten(), dtype=torch.bool).to(self.args.device) # (H*W,)
        return mask # (H*W,)  
    
