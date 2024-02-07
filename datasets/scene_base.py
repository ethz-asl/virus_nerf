import numpy as np
from abc import abstractmethod

from args.args import Args


class SceneBase():
    def __init__(
        self, 
        args:Args
    ):
        """
        Class to handle robot@home2 scenes. The scene's point cloud serves as ground truth.
        Args:
            args: arguments; Args object
        """ 
        self.args = args

        # scene transformation variables: world (in meters) -> cube ([-0.5,0.5]**3) coordinate system
        self.w2c_params = {
            "defined": False,
            "shift": None,
            "scale": None,
            "cube_min": -0.5,
            "cube_max": 0.5,
            "scale_margin": 1.05,
        }

        self._point_cloud = self._loadPointCloud()

    @abstractmethod
    def _defineParams(self):
        pass

    @abstractmethod
    def _loadPointCloud(self):
        pass

    def getPointCloud(
        self,
    ):
        """
        Get point cloud of scene.
        Returns:
            point cloud of scene; numpy array (N, x y z R B G)
        """       
        return self._point_cloud

    def getSliceMap(
        self, 
        height:float, 
        res:int, 
        height_tolerance:float, 
        height_in_world_coord:bool=True
    ):
        """
        Get a map of the scene of size (res,res). The map is a slice of the scene 
        at height [height-height_tolerance, height+height_tolerance].
        Args:
            height: height of the slice in world coordinates; float
            res: size of slice map; int
            height_tolerance: tolerance for height in world coordinates; float
            height_in_world_coord: if True, height is given in world otherwise in cube coordinates; bool
        Returns:
            slice_map: slice of the scene; array of shape (res, res)
        """
        slice_map = np.zeros((res, res)) # (x y)
        point_cloud = np.copy(self._point_cloud[:,:3]) # (N, x y z)

        if not height_in_world_coord:
            height = self.c2w(pos=np.array([[0,0,height]]))[0,2]

        # extract points in slice [height-height_tolerance, height+height_tolerance]
        point_idxs = np.where((point_cloud[:,2] >= height-height_tolerance) 
                              & (point_cloud[:,2] <= height+height_tolerance))[0] # (M,)
        points = point_cloud[point_idxs,:2] # (M, x y z)  

        # convert points to slice map indices ([0,res-1]**2)
        map_idxs = self.w2idx(pos=points, res=res) # (M, x y)

        # fill slice map
        slice_map[map_idxs[:,0], map_idxs[:,1]] = 1

        return slice_map
    
    def getSliceScan(
        self, 
        res:int, 
        rays_o:np.array, 
        rays_d:np.array=None, 
        height_tolerance:float=0.1, 
        angular_range:list=[-np.pi,np.pi], 
        rays_o_in_world_coord:bool=True
    ):
        """
        Get a scan of the scene at points. The scan is similar to a horizontal 2D LiDAR scan and represents
        the distance to the closest point in the scene in the direction of the scan.
        Args:
            res: size of slice map; int
            rays_o: origin of scan rays; numpy array of shape (N, 2/3)
            rays_d: direction of scan rays; numpy array of shape (N, 2/3)
            height_tolerance: tolerance for height in world coordinate system; float
            angular_res: size of scan / nb. angles per scan; int
            angular_range: range of scan angles ([min, max] where min is inclusive and max is exclusive); list of two floats
            rays_o_in_world_coord: if True, rays_o is given in world otherwise in cube coordinates; bool
        Returns:
            scan_map: scan of the scene at rays_o; numpy array of shape (res, res)
            scan_depth: distance to closest point in scene at rays_o; numpy array of shape (N,)
            scan_ray_angles: angles of scan rays; numpy array of shape (N,)
        """
        if rays_o_in_world_coord:
            rays_o = self.w2c(pos=rays_o, copy=True) # (N, 2)

        # calculate scan rays in cube coordinate system
        scan_rays_c, scan_angles = self._calcScanRays(res, rays_o=rays_o, rays_d=rays_d, angular_range=angular_range) # (N*M, 2)

        # verify that all height values are inside tolerance
        height_mean = np.mean(rays_o[:,2])
        if np.any(np.abs(rays_o[:,2] - height_mean) > height_tolerance):
            self.args.logger.error(f"height values of rays_o are not inside tolerance of {height_tolerance}!" \
                    + f" mean height: {height_mean}, min height: {np.min(rays_o[:,2])}, max height: {np.max(rays_o[:,2])}")

        # get slice map for mean height
        slice_map = self.getSliceMap(height_mean, res, height_tolerance=height_tolerance, height_in_world_coord=False) # (res, res)

        # get occupancy status of points on rays
        scan_rays_idxs = self.c2idx(pos=scan_rays_c, res=res) # (N*M, 2)
        scan_rays_occ = slice_map[scan_rays_idxs[:,0], scan_rays_idxs[:,1]] # (N*M,)

        scan_rays_idxs = scan_rays_idxs.reshape((rays_o.shape[0], -1, 2)) # (N, M, 2)
        scan_rays_occ = scan_rays_occ.reshape((rays_o.shape[0], -1)) # (N, M)

        # get closest occupied point on rays
        angle_idxs, point_idxs = np.where(scan_rays_occ > 0) # (occ_points,)
        angle_idxs, unique_ixds = np.unique(angle_idxs, return_index=True) # (closest_occ_points,)
        point_idxs = point_idxs[unique_ixds] # (closest_occ_points,)
        scan_rays_closest_idxs = scan_rays_idxs[angle_idxs, point_idxs] # (closest_occ_points, 2)
        
        # create scan map where all closest occupied points on ray are set to 1
        scan_map = np.zeros((res, res)) # (res, res)
        scan_map[scan_rays_closest_idxs[:,0], scan_rays_closest_idxs[:,1]] = 1

        # create scan depth that represents the distance in cube coordinates to the closest occupied point on ray
        scan_depth = np.full(rays_o.shape[0], np.nan) # (N,)
        scan_rays_closest_c = self.idx2c(map_idxs=scan_rays_closest_idxs, res=res) # (closest_occ_points, 2)
        scan_depth[angle_idxs] = np.linalg.norm(scan_rays_closest_c - rays_o[angle_idxs,:2], axis=1) # (closest_occ_points,)

        return scan_map, scan_depth, scan_angles
    
    def w2c(
        self, 
        pos, 
        only_scale:bool=False, 
        copy:bool=True
    ):
        """
        Transformation from world (in meters) to cube ([cube_min,cube_max]**3) coordinate system.
        Args:
            pos: position to scale and shift; tensor or array of shape (N, 2/3)
            only_scale: if True, only scale position and do not shift; bool
            copy: if True, copy pos before transformation; bool
        Returns:
            pos: scaled and shifted position; tensor or array of shape (N, 2/3)
        """
        if copy:
            pos = np.copy(pos)
        
        if not self.w2c_params["defined"]:
            self._defineParams()

        if not only_scale:
            pos -= self.w2c_params["shift"][:pos.shape[1]]
        pos /=  self.w2c_params["scale"]
        return pos
    
    def c2w(
        self, 
        pos, 
        only_scale:bool=False, 
        copy:bool=True
    ):
        """
        Transformation from cube ([cube_min,cube_max]**3) to world (in meters) coordinate system.
        Args:
            pos: position to scale and shift; tensor or array of shape (N, 2/3)
            only_scale: if True, only scale position and do not shift; bool
            copy: if True, copy pos before transformation; bool
        Returns:
            pos: scaled and shifted position; tensor or array of shape (N, 2/3)
        """
        if copy:
            pos = np.copy(pos)

        if not self.w2c_params["defined"]:
            self._defineParams()

        pos *=  self.w2c_params["scale"]
        if not only_scale:
            pos += self.w2c_params["shift"][:pos.shape[1]]
        return pos
    
    def c2idx(
        self, 
        pos, 
        res:int
    ):
        """
        Transformation from cube coordinate system ([cube_min,cube_max]**2) to slice map indices ([0,res-1]**2).
        Args:
            pos: position to transform; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
        """
        map_idxs = (res - 1) * (pos - self.w2c_params["cube_min"]) \
                / (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]) # (N, x y)
        return np.clip(map_idxs.round().astype(int), 0, res-1) # convert to int
    
    def idx2c(
        self, 
        map_idxs, 
        res:int
    ):
        """
        Transformation from slice map indices ([0,res-1]**2) to cube coordinate system ([cube_min,cube_max]**2).
        Args:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            pos: position in cube coordinate system; tensor or array of shape (N, 2)
        """
        pos = (map_idxs * (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]) / (res - 1)) \
                + self.w2c_params["cube_min"] # (N, x y)
        return pos
    
    def w2idx(
        self, 
        pos, 
        res:int
    ):
        """
        Transformation from world (in meters) to slice map indices ([0,res-1]**2).
        Args:
            pos: position to transform; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
        """
        pos_c = self.w2c(pos=pos)
        return self.c2idx(pos=pos_c, res=res)
    
    def idx2w(
        self, 
        map_idxs, 
        res:int
    ):
        """
        Transformation from slice map indices ([0,res-1]**2) to world (in meters).
        Args:
            map_idxs: slice map indices; tensor or array of shape (N, 2)
            res: size of slice map; int
        Returns:
            pos_w3: position in world coordinate system; tensor or array of shape (N, 2)
        """
        pos_c = self.idx2c(map_idxs=map_idxs, res=res)
        return self.c2w(pos=pos_c)
    
    def pos2map(
        self,
        pos:np.ndarray,
        num_points:int,
    ):
        """
        Convert position to slice map.
        Args:
            pos: position to transform; array of shape (N*M, 2)
            num_points: number of points in slice map; int
        Returns:
            map: map; array of shape (N, L, L)
        """
        idxs = self.w2idx(
            pos=pos, 
            res=self.args.eval.res_map,
        ) # (N*M, 2)
        idxs = idxs.reshape((num_points, -1, 2))

        map = np.zeros((num_points, self.args.eval.res_map, self.args.eval.res_map))
        map[np.arange(num_points)[:,None,None], idxs[:,:,0], idxs[:,:,1]] = 1
        return map
    
    def depth2pos(
        self, 
        rays_o:np.array, 
        rays_d:np.array, 
        depths:np.array
    ):
        """
        Convert depth values to positions.
        Args:
            rays_o: origin of scan rays in cube coordinates; numpy array of shape (N, 3)
            rays_d: direction of scan rays; numpy array of shape (N, 3)
            depths: depth values of scan rays; numpy array of shape (N,)
        Returns:
            pos: position of scan rays; numpy array of shape (N, 2)
            pos_o: origin of scan rays in plane; numpy array of shape (N, 2)
            dists: distance of scan rays; numpy array of shape (N,)
        """
        # valid mask
        mask = (~np.isnan(depths)) & np.all(~np.isnan(rays_o), axis=1) & np.all(~np.isnan(rays_d), axis=1)

        # collapse rays from 3D to 2D
        pos_o_temp, angles, dists_temp = self.space2plane(
            rays_o=rays_o[mask],
            rays_d=rays_d[mask],
            depths=depths[mask],
        )
        pos_temp = pos_o_temp + np.stack((dists_temp * np.cos(angles), dists_temp * np.sin(angles)), axis=1)

        # recover nan values
        pos = np.full((mask.shape[0], 2), np.nan)
        pos_o = np.full((mask.shape[0], 2), np.nan)
        dists = np.full(mask.shape[0], np.nan)
        pos[mask] = pos_temp
        pos_o[mask] = pos_o_temp
        dists[mask] = dists_temp
        return pos, pos_o, dists
    
    def space2plane(
        self,
        rays_o:np.array,
        rays_d:np.array,
        depths:np.array,
    ):
        """
        Convert 3D rays to 2D rays. Verify that all rays are in the same plane.
        Args:
            rays_o: origin of scan rays; numpy array of shape (N, 3)
            rays_d: direction of scan rays; numpy array of shape (N, 3)
            depths: depth values of scan rays; numpy array of shape (N,)
        Returns:
            pos_o: origin of scan rays in plane; numpy array of shape (N, 2)
            angles: angles of scan rays; numpy array of shape (N,)
            dist: distance of scan rays; numpy array of shape (N,)
        """
        rays_o = np.copy(rays_o) # (N, 3)
        rays_d = np.copy(rays_d) # (N, 3)
        depths = np.copy(depths) # (N,)

        if self.args.training.debug_mode:
            # verify that all rays_d have norm 1
            if not np.allclose(np.linalg.norm(rays_d, axis=1), 1):
                self.args.logger.error(f"norm of rays_d is not 1: norm = {np.linalg.norm(rays_d[:,:2], axis=1)}!")
                self.args.logger.error(f"norm min = {np.min(np.linalg.norm(rays_d[:,:2], axis=1))}, "
                                       +f"norm max = {np.max(np.linalg.norm(rays_d[:,:2], axis=1))}")

        # convert rays from 3D to 2D
        pos_o = rays_o[:,:2] # (N, 2)
        dist = depths * np.linalg.norm(rays_d[:,:2], axis=1) # (N,)
        angles = self.direction2angle(
            rays_d=rays_d[:,:2]
        ) # (N,)

        return pos_o, angles, dist
    
    def direction2angle(
        self,
        rays_d:np.array
    ):
        """
        Convert direction to angles.
        Args:
            rays_d: direction of scan rays; numpy array of shape (N, 2)
        Returns:
            angles: angles of scan rays; numpy array of shape (N,)
        """
        if self.args.training.debug_mode:
            # verify that all rays_d has only 2 dimensions
            if rays_d.shape[1] != 2:
                self.args.logger.error(f"rays_d has not only 2 dimensions!")
        
        angles = np.arctan2(rays_d[:,1], rays_d[:,0]) # (N,)
        return angles

    def _calcScanRays(
        self, 
        res:int, 
        rays_o:np.array, 
        rays_d:np.array=None, 
        angular_range:list=[-np.pi,np.pi]
    ):
        """
        Calculate scan rays in cube coordinate system. If rays_d is not given (None), the angles are linearly spaced 
        in the angular range.
        Args:
            res: size of slice map; int
            rays_o: origin of scan rays in cube coordinates; numpy array of shape (N, 3)
            rays_d: direction of scan rays; numpy array of shape (N, 3)
            angular_range: range of scan angles ([min, max] where min is inclusive and max is exclusive); list of two floats
        Returns:
            scan_rays_c: scan rays in cube coordinate system; numpy array of shape (N*M, 2)
            scan_angles: angles of scan rays; numpy array of shape (N,)
        """
        rays_o = np.copy(rays_o[:,:2]) # (N, 2)

        # if rays_d is None, linearly space angles between in angular_range
        # else, use given ray directions
        if rays_d is None:
            scan_angles = np.linspace(angular_range[0], angular_range[1], rays_o.shape[0], endpoint=False) # (N,)
        else:
            scan_angles = np.arctan2(rays_d[:,1], rays_d[:,0]) # (N,)

        # get relative scan rays
        M = np.ceil(np.sqrt(2*res**2)).astype(int)
        r_points = np.linspace(0, (self.w2c_params["cube_max"]-self.w2c_params["cube_min"]), M) # (M,)
        m_points, m_angles = np.meshgrid(r_points, scan_angles, indexing="xy") # (N, M)
        r_x = m_points * np.cos(m_angles) # (N, M)
        r_y = m_points * np.sin(m_angles) # (N, M)
        r_c = np.stack((r_x.flatten(), r_y.flatten()), axis=1) # (N*M, 2)

        # get absolute scan rays in cube coordinate system
        rays_o = np.repeat(rays_o, M, axis=0) # (N*M, 2)
        scan_rays_c = r_c + rays_o # (N*M, 2)
        return scan_rays_c, scan_angles
