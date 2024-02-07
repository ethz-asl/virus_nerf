import numpy as np
from abc import abstractmethod

from args.args import Args


class SensorModel():
    def __init__(
        self, 
        args:Args, 
        img_wh:tuple
    ) -> None:
        self.args = args
        self.W = img_wh[0]
        self.H = img_wh[1]
        
    @abstractmethod
    def convertDepth(self, depths, format):
        pass

    def pos2idx(
        self, 
        pos_h:np.array, 
        pos_w:np.array,
    ):
        """
        Convert position to index.
        Args:
            pos_h: position; array of shape (N,)
            pos_w: position; array of shape (N,)
        Returns:
            idxs_h: index; array of shape (N,)
            idxs_w: index; array of shape (N,)
        """
        idxs_h = None
        if pos_h is not None:
            idxs_h = np.round(pos_h).astype(int)
            idxs_h = np.clip(idxs_h, 0, self.H-1)

        idxs_w = None
        if pos_w is not None:
            idxs_w = np.round(pos_w).astype(int)
            idxs_w = np.clip(idxs_w, 0, self.W-1)

        return idxs_h, idxs_w

    def AoV2pixel(
        self, 
        aov_sensor:list
    ):
        """
        Convert the angle of view to width in pixels
        Args:
            aov_sensor: angle of view of sensor in width and hight; list
        Returns:
            num_pixels: width in pixels; int
        """
        img_wh = np.array([self.W, self.H])
        aov_sensor = np.array(aov_sensor)
        aov_camera = self.args.rgbd.angle_of_view

        num_pixels = img_wh * aov_sensor / aov_camera
        return np.round(num_pixels).astype(int)
    







