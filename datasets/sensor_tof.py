import numpy as np
import torch

from args.args import Args
from datasets.sensor_base import SensorModel


class ToFModel(SensorModel):
    def __init__(
        self, 
        args:Args, 
        img_wh:tuple
    ) -> None:
        """
        Sensor model for Time of Flight (ToF) sensor.
        Args:
            img_wh: image width and height, tuple of int
        """
        SensorModel.__init__(
            self, 
            args=args, 
            img_wh=img_wh,
        )    

        self.mask = self._createMask() # (H*W,)
        self.error_mask = self._createErrorMask(
            mask=self.mask.detach().clone().cpu().numpy(),
        ) # (H*W,)
        

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
                    "sensor": depth per ToF pixel; depths array of shape (N, 8*8)
        Returns:
            depths: depth img converted to ToF sensor array; array of shape (N, H*W)
        """
        depths = np.copy(depths) # (N, H*W)
        depths_out = np.zeros((depths.shape[0], self.H*self.W), dtype=np.float32) # (N, H*W)
        fov_mask = self.mask.detach().clone().cpu().numpy() # (H*W,)
        error_mask = self.error_mask.detach().clone().cpu().numpy() # (H*W,)

        if self.args.training.debug_mode:
            if np.any(depths == 0.0):
                self.args.logger.error(f"ToFModel.convertDepth: depths == 0.0")

        if format == "img":
            depths_out[:, fov_mask] = depths[:,error_mask] 
        elif format == "sensor":
            depths_out[:, fov_mask] = depths
        else:
            self.args.logger.error(f"Unknown depth format: {format}")

        # dilate depth img
        if self.args.tof.tof_pix_size > 1:
            depths_out = depths_out.reshape(depths.shape[0], self.H, self.W) # (N, H, W)
            depths_out = grey_dilation(depths_out, size=(1,self.args.tof.tof_pix_size,self.args.tof.tof_pix_size)) # (N, H, W)
            depths_out = depths_out.reshape(depths.shape[0], -1) # (N, H*W)
        depths_out[depths_out == 0.0] = np.nan # (N, H*W)  

        if (self.args.tof.sensor_random_error == 0.0) or (self.args.tof.sensor_random_error is None):
            return depths_out
        
        # add random error to depths
        self.args.logger.info(f"Add random error to ToF depths: {self.args.tof.sensor_random_error}°")
        valid_depths = ~np.isnan(depths_out) # (N, H*W)
        rand_error = np.random.normal(loc=0.0, scale=self.args.tof.sensor_random_error, size=depths_out.shape) # (N, H*W)
        depths_out[valid_depths] += rand_error[valid_depths]
        return depths_out
    
    def _createMask(
        self,
    ):
        """
        Create mask for ToF sensor.
        Returns:
            mask: mask for ToF sensor; tensor of shape (H*W,)
        """
        # calculate indices of ToF sensor array
        pix_wh = self.AoV2pixel(aov_sensor=self.args.tof.angle_of_view)
        idxs_w = np.linspace(0, pix_wh[0], self.args.tof.matrix[0], dtype=float)
        idxs_h = np.linspace(0, pix_wh[1], self.args.tof.matrix[1], dtype=float)

        # ajust indices to quadratic shape
        idxs_w = idxs_w + (self.W - pix_wh[0])/2
        idxs_h = idxs_h + (self.H - pix_wh[1])/2

        # convert indices to ints
        idxs_h, idxs_w = self.pos2idx(idxs_h, idxs_w) # (H,), (W,)     

        # create meshgrid of indices
        idxs_h, idxs_w = np.meshgrid(idxs_h, idxs_w, indexing='ij') # (H, W)
        self.idxs_h = idxs_h.flatten() # (H*W,)
        self.idxs_w = idxs_w.flatten() # (H*W,)

        # create mask
        mask = np.zeros((self.H, self.W), dtype=bool) # (H, W)
        mask[idxs_h, idxs_w] = True
        mask = torch.tensor(mask.flatten(), dtype=torch.bool).to(self.args.device) # (H*W,)
        return mask # (H*W,)
    
    def _createErrorMask(
        self,
        mask:torch.Tensor,
    ):
        """
        Create error mask for ToF sensor. If the calibration error is equal to 0.0, 
        the error mask is equal to the mask. Otherwise, the error mask is a shifted
        in a random direction by the calibration error. In this case, the ToF-depth is
        evaluated by using the error mask and assigned to the pixel in the mask.
        Args:
            mask: mask for ToF sensor; tensor of shape (H*W,)
        Returns:
            error_mask: error mask for ToF sensor; tensor of shape (H*W,)
        """
        mask = np.copy(mask) # (H*W,)
        if self.args.tof.sensor_calibration_error == 0.0:
            return torch.tensor(mask, dtype=torch.bool).to(self.args.device)

        # determine error in degrees
        direction = 0.0
        error = self.args.tof.sensor_calibration_error * np.array([np.cos(direction), np.sin(direction)]).flatten()

        # convert error to pixels
        error[0] = self.H * error[0] / self.args.rgbd.angle_of_view[0]
        error[1] = self.W * error[1] / self.args.rgbd.angle_of_view[1]
        error = np.round(error).astype(int)

        # convert error to mask indices
        mask = mask.reshape(self.H, self.W)
        idxs = np.argwhere(mask)
        idxs[:,0] = np.clip(idxs[:,0] + error[0], 0, self.H-1)
        idxs[:,1] = np.clip(idxs[:,1] + error[1], 0, self.W-1)

        # apply error to mask
        error_mask = np.zeros((self.H, self.W), dtype=bool)
        error_mask[idxs[:,0], idxs[:,1]] = True
        error_mask = torch.tensor(error_mask.flatten(), dtype=torch.bool).to(self.args.device)
        return error_mask # (H*W,)