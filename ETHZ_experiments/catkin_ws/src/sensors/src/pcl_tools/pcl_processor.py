#!/usr/bin/env python
import numpy as np


class PCLProcessor:
    def __init__(
        self,
    ):
        pass
        
    def limitXYZ(
        self,
        xyzi:np.array,
        x_lims:tuple=None,
        y_lims:tuple=None,
        z_lims:tuple=None,
    ):
        """
        Limit the pointcloud to a certain range in x, y and z.
        Args:
            xyzi: pointcloud; numpy array (N,4)
            x_lims: limits in x; tuple of length 2
            y_lims: limits in y; tuple of length 2
            z_lims: limits in z; tuple of length 2
        Returns:
            xyz: limited pointcloud; numpy array (N,3)
        """
        if x_lims is not None:
            xyzi = xyzi[np.logical_and(xyzi[:,0] >= x_lims[0], xyzi[:,0] <= x_lims[1])]
        if y_lims is not None:
            xyzi = xyzi[np.logical_and(xyzi[:,1] >= y_lims[0], xyzi[:,1] <= y_lims[1])]
        if z_lims is not None:
            xyzi = xyzi[np.logical_and(xyzi[:,2] >= z_lims[0], xyzi[:,2] <= z_lims[1])]
        return xyzi
    
    def limitRTP(
        self,
        xyzi,
        r_lims:tuple=None,
        t_lims:tuple=None,
        p_lims:tuple=None,
    ):
        """
        Limit the pointcloud to a certain range in radius, theta and phi.
        Args:
            xyzi: pointcloud; numpy array (N,4)
            r_lims: limits in radius; tuple of length 2
            t_lims: limits in theta [degrees]; tuple of length 2
            p_lims: limits in phi [degrees]; tuple of length 2
        Returns:
            xyz: limited pointcloud; numpy array (N,3)
        """
        if r_lims is None and t_lims is None and p_lims is None:
            return xyzi
        
        rtpi = self._cart2sph(
            xyzi=xyzi
        )
        
        if r_lims is not None:
            rtpi = rtpi[np.logical_and(rtpi[:,0] >= r_lims[0], rtpi[:,0] <= r_lims[1])]
        if t_lims is not None:
            t_lims = np.deg2rad(t_lims)
            rtpi = rtpi[np.logical_and(rtpi[:,1] >= t_lims[0], rtpi[:,1] <= t_lims[1])]
        if p_lims is not None:
            p_lims = np.deg2rad(p_lims)
            rtpi = rtpi[np.logical_and(rtpi[:,2] >= p_lims[0], rtpi[:,2] <= p_lims[1])]
        
        xyzi = self._sph2cart(
            rtpi=rtpi,
        )
        return xyzi
        
    def offsetDepth(
        self,
        xyzi:np.array,
        offset:float,
    ):
        """
        Offsets the depth of a pointcloud assuming sensor is at position (0,0,0).
        Coordinate system: LiDAR
            x: points in the viewing direction
            y: points to the left
            z: points upwards
        Args:
            xyzi: pointcloud; numpy array (N,4)
            offset: offset in ray direction; float
        Returns:
            xyzi: pointcloud with offset; numpy array (N,4)
        """
        if offset is None:
            return xyzi
        
        rtpi = self._cart2sph(
            xyzi=xyzi
        )
        
        rtpi[:,0] += offset
        
        xyzi = self._sph2cart(
            rtpi=rtpi
        )
        return xyzi

    def _cart2sph(
        self,
        xyzi
    ):
        """
        Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi).
        Args:
            xyzi: points in cartesian coordinates; numpy array (N,4)
        Returns:
            rtpi: points in polar coordinates; numpy array (N,4)
        """
        radius = np.linalg.norm(xyzi[:,:3], axis=1)
        theta = np.arctan2(np.sqrt(xyzi[:,0]**2 + xyzi[:,1]**2), xyzi[:,2])
        phi = np.arctan2(xyzi[:,1], xyzi[:,0])
        return np.stack((radius, theta, phi, xyzi[:,3]), axis=1)

    def _sph2cart(
        self,
        rtpi,
    ):
        """
        Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z).
        Args:
            rtpi: points in polar coordinates; numpy array (N,4)
        Returns:
            xyzi: points in cartesian coordinates; numpy array (N,4)
        """
        x = rtpi[:,0] * np.cos(rtpi[:,2]) * np.sin(rtpi[:,1])
        y = rtpi[:,0] * np.sin(rtpi[:,2]) * np.sin(rtpi[:,1])
        z = rtpi[:,0] * np.cos(rtpi[:,1])
        return np.stack((x, y, z, rtpi[:,3]), axis=1)
        

if __name__ == '__main__':
    pass
