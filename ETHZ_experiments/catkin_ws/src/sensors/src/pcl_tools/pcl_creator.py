#!/usr/bin/env python
import numpy as np
import pandas as pd

from abc import abstractmethod
import os



class PCLCreator():
    def __init__(
        self,
    ):
        pass
    
    @abstractmethod 
    def meas2depth(
        self,
        meas:np.array,
    ):
        pass
    
    def meas2pcl(
        self,
        meas:np.array,
    ):
        """
        Convert depth measurments to pointcloud depending on sensor type.
        Args:
            meas: meas measurments; numpy array of shape (N,)
        Returns:
            xyz: pointcloud; numpy array of shape (N,3)
        """
        depth = self.meas2depth(
            meas=meas,
        )
        xyz = self.depth2pcl(
            depth=depth,
        )
        return xyz
    
    def depth2pcl(
        self,
        depth:np.array,
    ):
        """
        Converting depth into 3D point cloud.
        Args:
            depth: converted depth measurement; np.array (H, W)
        Returns:
            xyz: point cloud; np.array (H*W, 3)
        """
        depth = depth.reshape(-1, 1) # (H*W, 1)
        xyz = self.directions * depth # (H*W, 3)
        return xyz
    
    def fovDirections(
        self,
        fov_xy:list,
        W:int,
        H:int,
    ):
        """
        Calculate directions given a field of view.
        Coordinate system: camera
            x: points to the right
            y: points downwards
            z: points into the viewing direction
        Args:
            fov_xy: field of view in degrees (x and y directions); list of length 2
            W: width of image; int
            H: height of image; int
        Returns:
            directions: ray directions; numpy array of shape (H*W, 3)
        """
        fov_xy = np.deg2rad(fov_xy) # (2,)
        num_pts = np.array([W, H]) # (2,)
        
        fov_cells = fov_xy / num_pts
        angle_max = fov_cells * (num_pts - 1) / 2
        angle_min = - angle_max
        
        angles_x = np.linspace(angle_min[0], angle_max[0], num_pts[0]) # (W,)
        angles_y = np.linspace(angle_min[1], angle_max[1], num_pts[1]) # (H,)
        angles_x, angles_y = np.meshgrid(angles_x, angles_y, indexing="xy") # (H,W), (H,W)
        angles_x = angles_x.flatten() # (H*W,)
        angles_y = angles_y.flatten() # (H*W,)
        
        x = np.sin(angles_x) # (H*W,)
        y = np.sin(angles_y) # (H*W,)
        z = np.cos(angles_x) * np.cos(angles_y) # (H*W,)
        directions = np.stack((x, y, z), axis=1) # (H*W, 3)
        
        return directions
    
    def cameraDirections(
        self,
        fx:float,
        fy:float,
        cx:float,
        cy:float,
        W:int,
        H:int,
    ):
        """
        Calculate directions given focal lengths of a camera.
        Coordinate system: upsdie-down camera
            x: points to the left
            y: points upwards
            z: points into the viewing direction
        Args:
            fx: focal length in x direction; float
            fy: focal length in y direction; float
            cx: center of projection in x direction; float
            cy: center of projection in y direction; float
            W: width of image; int
            H: height of image; int
        Returns:
            directions: ray directions; numpy array of shape (H*W, 3)
        """
        us, vs = np.meshgrid(np.arange(W), np.arange(H), indexing="xy") # (H,W), (H,W)
        dir_x = (us - cx + 0.5) / fx # (H,W)
        dir_y = (vs - cy + 0.5) / fy # (H,W)
        dir_z = np.ones_like(us) # (H,W)
        
        directions = np.stack((dir_x, dir_y, dir_z), axis=2) # (H,W,3)
        directions /= np.linalg.norm(directions, axis=2, keepdims=True) # (H,W,3)
        directions = directions.reshape(-1, 3) # (H*W, 3)
        
        return directions
        
        
class PCLCreatorUSS(PCLCreator):
    def __init__(
        self,
        W:int,
        H:int,
    ):
        super().__init__()
        
        self.W = W
        self.H = H
        self.directions = self.fovDirections(
            fov_xy=[55, 35],
            W=self.W,
            H=self.H,
        )
        
    def meas2depth(
        self,
        meas:float,
    ):
        """
        Convert depth measurments to meters and filter false measurments.
        Args:
            meas: depth measurments; float
        Returns:
            depth: depth measurments; np.array of floats (H, W)
        """
        if meas >= 50000:
            meas = 0.0
        depth = meas / 5000
        return depth * np.ones((self.H, self.W))

    
class PCLCreatorToF(PCLCreator):
    def __init__(
        self,
        H:int,
        W:int,
     ):
        super().__init__()
        
        self.directions = self.fovDirections(
            fov_xy=[45, 45],
            W=W,
            H=H,
        )
        
        self.depth_min = 0.1
        
    def meas2depth(
        self,
        meas:float,
    ):
        """
        Convert depth measurments to meters and correct reference frame.
        Args:
            meas: depth measurments; tuple of floats (64,)
        Returns:
            depth: depth measurments; np.array of floats (8, 8)
        """
        meas = np.array(meas, dtype=np.float32)
        depth = 0.001 * meas
        
        depth[depth <= self.depth_min] = np.nan
        
        depth = depth.reshape(8, 8)
        depth = depth[:, ::-1].T
        depth = depth[::-1, ::-1]
        return depth
    

class PCLCreatorRS(PCLCreator):
    def __init__(
        self,
        data_dir:str,
        sensor_id:str,
    ):
        super().__init__()
        
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(data_dir, "../camera_intrinsics.CSV"),
            dtype={'cam_id': str, 'fx': np.float64, 'fy': np.float64, 'cx': np.float64, 'cy': np.float64},
        )
        df_cam = df[df["cam_id"] == sensor_id]
        
        self.directions = self.cameraDirections(
            fx=df_cam['fx'].values[0],
            fy=df_cam['fy'].values[0],
            cx=df_cam['cx'].values[0],
            cy=df_cam['cy'].values[0],
            W=640,
            H=480,
        )

    def meas2depth(
        self,
        meas:float,
    ):
        """
        Convert depth measurments to meters and correct reference frame.
        Args:
            meas: depth measurments; np.array of floats (H, W)
        Returns:
            depth: depth measurments; np.array of floats (H, W)
        """
        meas = np.array(meas, dtype=np.float32) # (H, W)
        H = meas.shape[0]
        W = meas.shape[1]
        depth = 0.001 * meas # (H, W)    

        depth = depth.flatten() # (H*W,)
        depth = depth / self.directions[:,2] # (H*W,)
        depth = depth.reshape(H, W)
        
        return depth
    
    
    
    
def test_meas2depth():
    
    pcl_creator = PCLCreatorToF()
    
    depth = np.zeros((8,8))
    depth[0,0] = 1
    depth[0,-1] = 1
    
    xyz = pcl_creator.depth2pcl(
        depth=depth,
    )
    
    print(xyz)
    
if __name__ == "__main__":
    test_meas2depth()
    




    # def convert_depth_frame_to_pointcloud(
    #     self,
    #     depth_image, 
    #     fx,
    #     fy,
    #     cx,
    #     cy,
    #     H,
    #     W,
    # ):
    #     """
    #     Convert the depthmap to a 3D point cloud

    #     Parameters:
    #     -----------
    #     depth_frame 	 	 : rs.frame()
    #                         The depth_frame containing the depth map
    #     camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

    #     Return:
    #     ----------
    #     x : array
    #         The x values of the pointcloud in meters
    #     y : array
    #         The y values of the pointcloud in meters
    #     z : array
    #         The z values of the pointcloud in meters

    #     """
        
    #     [height, width] = depth_image.shape

    #     nx = np.linspace(0, width-1, width)
    #     ny = np.linspace(0, height-1, height)
    #     u, v = np.meshgrid(nx, ny)
    #     x = (u.flatten() - cx)/fx
    #     y = (v.flatten() - cy)/fy

    #     z = depth_image.flatten() / 1000;
    #     x = np.multiply(x,z)
    #     y = np.multiply(y,z)

    #     x = x[np.nonzero(z)]
    #     y = y[np.nonzero(z)]
    #     z = z[np.nonzero(z)]
    #     xyz = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)), axis=1)

    #     return xyz
    
    

# import torch
# from kornia import create_meshgrid
# @torch.cuda.amp.autocast(dtype=torch.float32)
# def get_ray_directions(H,
#                        W,
#                        K,
#                        device='cpu',
#                        random=False,
#                        return_uv=False,
#                        flatten=True):
#     """
#     Get ray directions for all pixels in camera coordinate [right down front].
#     Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
#                ray-tracing-generating-camera-rays/standard-coordinate-systems

#     Inputs:
#         H, W: image height and width
#         K: (3, 3) camera intrinsics
#         random: whether the ray passes randomly inside the pixel
#         return_uv: whether to return uv image coordinates

#     Outputs: (shape depends on @flatten)
#         directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
#         uv: (H, W, 2) or (H*W, 2) image coordinates
#     """
#     grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2)
#     u, v = grid.unbind(-1)

#     fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
#     if random:
#         directions = \
#             torch.stack([(u-cx+torch.rand_like(u))/fx,
#                          (v-cy+torch.rand_like(v))/fy,
#                          torch.ones_like(u)], -1)
#     else:  # pass by the center
#         directions = \
#             torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
#     if flatten:
#         directions = directions.reshape(-1, 3)
#         grid = grid.reshape(-1, 2)

#     if return_uv:
#         return directions, grid
#     return directions