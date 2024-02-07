import numpy as np
from robotathome import RobotAtHome

from args.args import Args
from datasets.scene_base import SceneBase

class SceneRH(SceneBase):
    def __init__(
        self, 
        rh:RobotAtHome, 
        args:Args
    ):
        """
        Class to handle robot@home2 scenes. The scene's point cloud serves as ground truth.
        Args:
            rh: robot@home2 database; RobotAtHome object
            args: arguments; Args object
        """ 
        self.rh = rh

        SceneBase.__init__(
            self, 
            args=args
        )
  
    def _loadPointCloud(self):
        """
        Load scene from robot@home2 database.
        Returns:
            point cloud of scene; numpy array (N, x y z R B G)
        """
        home_session_id = self.rh.name2id(self.args.rh.home+"-"+self.args.rh.home_session,'hs')
        room_id = self.rh.name2id(self.args.rh.home+"_"+self.args.rh.room, "r")

        # get scene database of particular room  
        scene =  self.rh.get_scenes().query(f'home_session_id=={home_session_id} & room_id=={room_id}')
    
        # load scene point cloud
        scene_file = scene.scene_file.values[0]
        return np.loadtxt(scene_file, skiprows=6)
    
    def _defineParams(self):
        """
        Calculate world (in meters) to cube ([cube_min,cube_max]**3) transformation parameters.
        Enlarge the cube with scale_margin s.t. all points are sure to be inside the cube.
        """
        point_cloud = np.copy(self._point_cloud[:,:3]) # (N, x y z)

        # get scene shift and scale
        xyz_min = point_cloud.min(axis=0)
        xyz_max = point_cloud.max(axis=0)

        shift = (xyz_max + xyz_min) / 2
        scale = (xyz_max - xyz_min).max() * self.w2c_params["scale_margin"] \
                / (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]) 

        # set world to cube transformation parameters
        self.w2c_params["defined"] = True
        self.w2c_params["shift"] = shift
        self.w2c_params["scale"] = scale
