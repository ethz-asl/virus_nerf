#!/usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation

class PCLTransformer():
    def __init__(
        self,
        T:np.array=None,
        t:np.array=None,
        q:np.array=None,
        R:np.array=None,
        euler_angles:np.array=None,
        euler_seq:str=None,
    ) -> None:
        self._t = None
        self._r = None
        self.setTransform(
            T=self._convert2Numpy(T),
            t=self._convert2Numpy(t),
            q=self._convert2Numpy(q),
            R=self._convert2Numpy(R),
            euler_angles=self._convert2Numpy(euler_angles),
            euler_seq=euler_seq,
        )
        
    def setTransform(
        self,
        T:np.array=None,
        t:np.array=None,
        q:np.array=None,
        R:np.array=None,
        euler_angles:np.array=None,
        euler_seq:str=None,
    ):
        """
        Set the transform.
        Args:
            T: homogenous matrix transformation; numpy array (4,4)
            t: translation; numpy array (3,)
            q: quaternion; numpy array (4,)
            R: rotation matrix; numpy array (3,3)
            euler_angles: euler angles [roll, pitch, yaw]; numpy array (3,)
            euler_seq: euler sequence; str
                (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)
        """
        if T is not None:
            self._r = Rotation.from_matrix(np.copy(T[:3,:3]))
            self._t = np.copy(T[:3,3])
            return
        
        if t is not None and q is not None:
            self._r = Rotation.from_quat(np.copy(q))
            self._t = np.copy(t)
            return
            
        if t is not None and R is not None:
            self._r = Rotation.from_matrix(np.copy(R))
            self._t = np.copy(t)
            return
        
        if t is not None and euler_angles is not None and euler_seq is not None:
            self._r = Rotation.from_euler(
                seq=euler_seq,
                angles=np.copy(euler_angles),
            )
            self._t = np.copy(t)
            return
            
        print(f"ERROR: PointCloudTransformer.setT: Transform is not complete")

    def getTransform(
        self,
        type:str,
    ):
        """
        Get transformation

        Args:
            type: type of transformation; str
                "matrix": return homogenous matrix transformation; numpy array (4,4)
                "quaternion": return quaternion and translation; numpy array (4,), numpy array (3,)
        Returns:
            T: transformation
        """
        if type == "matrix":
            T = np.eye(4)
            T[:3,:3] = Rotation.as_matrix(self._r)
            T[:3,3] = self._t
            return np.copy(T)
        
        if type == "quaternion":
            q = self._r.as_quat()
            t = self._t
            return np.copy(q), np.copy(t)
        
        print(f"ERROR: Transform.getT: type={type} not implemented")
            
    def invertTransform(
        self,
    ):
        """
        Invert trasnformation.
        """
        r_inv = self._r.inv()
        R_inv = Rotation.as_matrix(r_inv)
        t_inv = - R_inv @ self._t
        
        self._r = r_inv
        self._t = t_inv
        return self
    
    def concatTransform(
        self,
        add_transform,
        apply_first_add_transform:bool,
    ):
        """
        Concatenate additional with propre transformation.
        Args:
            additonal_transform: additional transformation; PCLTransformer
            apply_first_additonal: apply additional transformation first and then propre one; bool
        """
        T_self = self.getTransform(
            type="matrix",
        )
        T_add = add_transform.getTransform(
            type="matrix",
        )
        
        if apply_first_add_transform:
            T = T_self @ T_add
        else:
            T = T_add @ T_self
            
        self.setTransform(
            T=T,
        )
        return self
        
    def transformPointcloud(
        self,
        xyz:np.array,
    ):
        """
        Transform pointcloud.
        Args:
            xyz: pointcloud; numpy array (N,3)
        Returns:
            xyz: transformed pointcloud; numpy array (N,3)
        """
        N = xyz.shape[0]
        
        T = np.eye(4)
        T[:3,:3] = Rotation.as_matrix(self._r)
        T[:3,3] = self._t
        
        xyz1 = np.ones((4, N))
        xyz1[:3, :] = xyz.T
        
        xyz1 = T @ xyz1
        xyz = xyz1[:3,:].T
        return xyz
            
    def _convert2Numpy(
        self,
        obj,
    ):
        """
        Convert object to numpy array if not None.
        Args:
            obj: list
        Returns:
            obj: numpy array or None
        """
        if obj is None:
            return None
        return np.array(obj)



if __name__ == "__main__":
    pass