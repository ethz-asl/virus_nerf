import numpy as np

from args.args import Args
from datasets.sensor_base import SensorModel


class RGBDModel(SensorModel):
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
        SensorModel.__init__(self, args, img_wh)     

    def convertDepth(
        self, 
        depths:np.array,
        format:str="img",
    ):
        """
        Convert depth img using ToF sensor model. Set all unknown depths to nan.
        Args:
            depths: depth img; array of shape (N, H*W)
            format: not used
        Returns:
            depths: depth img converted to ToF sensor array; array of shape (N, H*W)
        """
        return np.copy(depths)