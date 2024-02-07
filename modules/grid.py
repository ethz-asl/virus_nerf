import torch
import numpy as np
import sys
import os
from abc import abstractmethod
 
sys.path.insert(0, os.getcwd())
from kornia.utils.grid import create_meshgrid3d
from modules.utils import (
    morton3D, 
    packbits, 
)
from args.args import Args


class Grid():
    def __init__(
        self,
        args:Args,
        grid_size:int,
        cascades:int,
        morton_structure:bool,
    ) -> None:
        self.args = args
        self.grid_size = grid_size
        self.cascades = cascades
        self.morton_structure = morton_structure

        self.grid_coords = create_meshgrid3d(
            self.grid_size, 
            self.grid_size, 
            self.grid_size, 
            False, 
            dtype=torch.int32
        ).reshape(-1, 3).to(device=self.args.device)

        self.bitfield = torch.zeros(self.cascades * self.grid_size**3 // 8, dtype=torch.uint8, device=self.args.device)

        if morton_structure:
            self.occ_morton_grid = torch.zeros(self.cascades, self.grid_size**3, device=self.args.device)
        else:
            self.occ_3d_grid = torch.zeros(self.grid_size, self.grid_size, self.grid_size)

    @abstractmethod
    def update(self):
        pass

    @torch.no_grad()
    def getBitfield(
        self,
        clone:bool=False,
    ):
        """
        Get bitfield.
        Args:
            clone: whether to clone the bitfield; bool
        Returns:
            grid: bitfield in morton layout; tensor of uint8 (grid_size**3 // 8,)
        """
        if clone:
            return self.bitfield.clone().detach()
        return self.bitfield
    
    @torch.no_grad()
    def getOccupancyCartesianGrid(
        self,
        clone:bool=False,
    ):
        """
        Get occupancy grid.
        Args:
            clone: whether to clone the grid; bool
        Returns:
            grid: cartesian occupancy grid; tensor (grid_size, grid_size, grid_size)
        """
        if self.morton_structure:
            grid = self.morton2cartesian(
                grid_morton=self.occ_morton_grid[0]
            )
        else:
            grid = self.occ_3d_grid
        
        if clone:
            return grid.clone().detach()
        return grid    

    @torch.no_grad()
    def getBinaryCartesianGrid(
        self,
        threshold:float,
    ):
        """
        Get binary cartesian grid.
        Args:
            threshold: threshold for occupancy grid; float
        Returns:
            grid: binary grid; tensor of bool (grid_size, grid_size, grid_size)
        """
        if self.morton_structure:
            grid = self.morton2cartesian(
                grid_morton=self.occ_morton_grid[0]
            )
        else:
            grid = self.occ_3d_grid
        
        return self.thresholdGrid(
            grid=grid,
            threshold=threshold,
        )

    @torch.no_grad()
    def getAllCells(
        self,
    ):
        """
        Get all cells from the density grid.
        Outputs:
            cells: list of tubles (of length self.cascades):
                    indices: morton indices; tensor of int32 (grid_size**3,)
                    coords: cartesian coordinates; tensor of int32 (grid_size**3, 3)
        """
        indices = morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells
    
    @torch.no_grad()
    def updateBitfield(
        self, 
        grid:torch.tensor,
        threshold:float,
        convert_cart2morton:bool,
    ):
        """
        Update bitfield with cartesian occupancy grid.
        Args:
            grid: cartesian occupancy grid; tensor of float32 (grid_size, grid_size, grid_size)
            threshold: threshold for occupancy grid; float
            convert_cart2morton: whether grid is cartesian and must be converted to morton structure first; bool
        """
        if convert_cart2morton:
            occ_morton = self.cartesian2morton(
                grid_3d=grid
            )
        else:
            occ_morton = grid
        
        self.bitfield = self.morton2bitfield(
            occ_morton=occ_morton, 
            threshold=threshold
        )
    
    @torch.no_grad()
    def cartesian2morton(
        self, 
        grid_3d:torch.tensor,
    ): 
        """
        Convert cartesian coordinates to morton indices.
        Args:
            grid_3d: 3D grid; tensor of either float32 or bool (grid_size, grid_size, grid_size)
        Returns:
            grid_morton: morton grid; tensor of either float32 or bool (grid_size**3,)
        """
        cells = self.getAllCells()
        indices, coords = cells[0]

        grid_morton = torch.zeros(self.grid_size**3, dtype=grid_3d.dtype, device=self.args.device)
        grid_morton[indices] = grid_3d[coords[:, 0], coords[:, 1], coords[:, 2]]
        return grid_morton
    
    @torch.no_grad()
    def morton2cartesian(
        self, 
        grid_morton:torch.tensor,
    ):
        """
        Convert morton indices to cartesian coordinates.
        Args:
            grid_morton: morton grid; tensor of either float32 or bool (grid_size**3,)
        Returns:
            grid_3d: 3D grid; tensor of either float32 or bool (grid_size, grid_size, grid_size)
        """
        cells = self.getAllCells()
        indices, coords = cells[0]

        grid_3d = torch.zeros(self.grid_size, self.grid_size, self.grid_size, dtype=grid_morton.dtype, device=self.args.device)
        grid_3d[coords[:, 0], coords[:, 1], coords[:, 2]] = grid_morton[indices]
        return grid_3d
    
    @torch.no_grad()
    def morton2bitfield(
        self, 
        occ_morton:torch.tensor, 
        threshold:float,
    ) -> torch.tensor:
        """
        Convert morton occupancy grid to morton bitfield.
        Args:
            occ_morton: morton occupancy grid; tensor of float32 (grid_size**3,)
            threshold: threshold for occupancy grid; float
        Returns:
            bin_bitfield: morton bitfield; tensor of uint8 (grid_size**3 // 8,)
        """
        bin_bitfield = torch.zeros(self.grid_size**3 // 8, dtype=torch.uint8, device=self.args.device)
        packbits(
            density_grid=occ_morton.reshape(-1).contiguous(),
            density_threshold=threshold,
            density_bitfield=bin_bitfield,
        )
        return bin_bitfield
    
    @torch.no_grad()
    def bitfield2morton(
        self, 
        bin_bitfield:torch.tensor,
    ):
        """
        Convert morton bitfield to morton binary grid.
        Args:
            bin_bitfield: morton bitfield; tensor of uint8 (grid_size**3 // 8,)
        Returns:
            bin_morton: morton binary grid; tensor of bool (grid_size**3,)
        """
        bin_bitfield = bin_bitfield.clone().detach()

        mask = torch.tensor([[1, 2, 4, 8, 16, 32, 64, 128]], dtype=torch.uint8, device=self.args.device)
        mask = mask.repeat(bin_bitfield.shape[0], 1)

        bin_bitfield = bin_bitfield.reshape(-1, 1).repeat(1, 8)
        bin_morton = (bin_bitfield & mask).to(dtype=torch.bool)
        bin_morton = bin_morton.reshape(-1)
        return bin_morton
    
    @torch.no_grad()
    def thresholdGrid(
        self, 
        grid:torch.tensor,
        threshold:float,
    ):
        """
        Threshold occupancy grid.
        Args:
            grid: occupancy grid; tensor of float32 (grid_size, grid_size, grid_size) or (grid_size**3,)
            threshold: threshold for occupancy grid; float
        Returns:
            grid_thr: thresholded occupancy grid; tensor of bool (same as grid)
        """
        grid_thr = torch.zeros_like(grid, dtype=torch.bool, device=self.args.device)
        grid_thr[grid <= threshold] = 0
        grid_thr[grid > threshold] = 1
        return grid_thr
    
    @torch.no_grad()
    def c2oCoordinates(
        self,
        pos_c,
    ):
        """
        Convert cube to occupancy grid coordinates.
        Args:
            pos_c: cube coordinates; tensor or array of float32
        Returns:
            pos_o: occupancy grid coordinates; tensor or array of int32
        """
        height_o = self.grid_size * (pos_c+self.args.model.scale) / (2*self.args.model.scale) 

        if torch.is_tensor(height_o):
            return torch.round(height_o).to(dtype=torch.int32)
        return np.round(height_o).astype(np.int32)