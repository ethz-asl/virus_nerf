import numpy as np
import torch
from alive_progress import alive_bar
from contextlib import nullcontext



def findNearestNeighbour(
    array1:np.ndarray, 
    array2:np.ndarray, 
    batch_size:int=None, 
    progress_bar:bool=False, 
    ignore_nan:bool=False,
):
    """
    Find the closest points in array2 for each point in array1
    and return the indices of array2 for each point in array1.
    Args:
        array1: array of float (N, 2/3)
        array2: array of float (M, 2/3)
        batch_size: batchify array1 to avoid memory error, if None, batch_size = N; int/None
        progress_bar: show progress bar; bool
        ignore_nan: ignore nan values in array2; bool
    Returns:
        nn_idxs: indices of nearest neighbours from array2 with respect to array1; array of int (N,)
        nn_dists: distances of nearest neighbours from array2 with respect to array1; array of float (N,)
    """
    # downsample arrays
    array1 = np.copy(array1.astype(np.float32))
    array2 = np.copy(array2.astype(np.float32))

    # remove nan values
    if ignore_nan:
        mask_array1 = ~(np.isnan(array1).any(axis=1))
        mask_array2 = ~(np.isnan(array2).any(axis=1))
        array1 = array1[mask_array1]
        array2 = array2[mask_array2]    

        if array1.shape[0] == 0 or array2.shape[0] == 0:
            nn_idxs = - np.ones(mask_array1.shape[0], dtype=np.int32)
            nn_dists = np.full(mask_array1.shape[0], np.nan, dtype=np.float32)
            return nn_idxs, nn_dists

    # define batch size
    if batch_size is None:
        batch_size = array1.shape[0]
    else:
        while array1.shape[0]%batch_size != 0:
            batch_size -= 1

    # determine nearest neighbour indices
    nn_idxs = - np.ones(array1.shape[0], dtype=np.int32) # (N,)
    with alive_bar(array1.shape[0]//batch_size, bar = 'bubbles', receipt=False) if progress_bar else nullcontext() as bar:

        # split calculation in batches to avoid memory error
        for i in range(0, array1.shape[0], batch_size):
            dist = np.linalg.norm(array2[:, np.newaxis] - array1[i:i+batch_size], axis=2) # (M, batch_size)
            nn_idxs[i:i+batch_size] = np.argmin(dist, axis=0).astype(np.int32) # (batch_size,)

            if progress_bar:
                bar()

    # determine nearest neighbour distances
    nn_dists = np.linalg.norm(array2[nn_idxs] - array1, axis=1) # (N,)

    # recover nan values
    if ignore_nan:
        nn_idxs_temp = - np.ones(mask_array1.shape[0], dtype=np.int32)
        nn_idxs_temp[mask_array1] = nn_idxs
        nn_idxs = nn_idxs_temp
        nn_dists_temp = np.full(mask_array1.shape[0], np.nan, dtype=np.float32)
        nn_dists_temp[mask_array1] = nn_dists
        nn_dists = nn_dists_temp
    
    return nn_idxs, nn_dists

def createScanRays(
    rays_o:np.ndarray,
    angle_res:int,
    angle_min_max:tuple=(-np.pi, np.pi),
):
    """
    Create scan rays for gievn ray origins.
    Args:
        rays_o: ray origins; array of shape (N, 3) or torch tensor of shape (N, 3)
        angle_res: number of angular samples (M); int
        angle_min_max: minimum and maximum angle (radians); tuple of float (2,)
    Returns:
        rays_o: ray origins; array of shape (N*M, 3) or torch tensor of shape (N*M, 3)
        rays_d: ray directions; array of shape (N*M, 3) or torch tensor of shape (N*M, 3)
    """
    ouput_is_tensor = False
    if isinstance(rays_o, torch.Tensor):
        ouput_is_tensor = True
        device = rays_o.device
        rays_o = rays_o.detach().clone().cpu().numpy()

    # create ray directions
    rays_d = np.zeros((angle_res, 3)) # (M, 3)
    angles = np.linspace(angle_min_max[0], angle_min_max[1], angle_res, endpoint=False)
    rays_d[:,0] = np.cos(angles)
    rays_d[:,1] = np.sin(angles)

    # repeat rays for different points and angles
    rays_d = np.tile(rays_d, (rays_o.shape[0], 1)) # (N*M, 3)
    rays_o = np.repeat(rays_o, angle_res, axis=0) # (N*M, 3)

    if ouput_is_tensor:
        rays_o = torch.tensor(rays_o, dtype=torch.float32, device=device)
        rays_d = torch.tensor(rays_d, dtype=torch.float32, device=device)
    return rays_o, rays_d

def createScanPos(
    res_map:int,
    height_c:float,
    num_avg_heights:int,
    tolerance_c:float,
    cube_min:float,
    cube_max:float,
    device:str,
):
    """
    Create map positions to evaluate density for different heights.
        N: number of points
        M: number of angular samples
        A: number of heights to average over
    Args:
        res_map: number of samples in each dimension (L); int
        height_c: height of slice in cube coordinates; float
        num_avg_heights: number of heights to average over (A); int
        tolerance_c: tolerance in cube coordinates; float
        cube_min: minimum cube coordinate; float
        cube_max: maximum cube coordinate; float
        device: device to use; str
    Returns:
        pos_avg: map positions for different heights; array of shape (L*L*A, 3)
    """
    # create map positions
    pos = torch.linspace(cube_min, cube_max, res_map).to(device) # (L,)
    m1, m2 = torch.meshgrid(pos, pos) # (L, L), (L, L)
    pos = torch.stack((m1.reshape(-1), m2.reshape(-1)), dim=1) # (L*L, 2)

    # create map positions for different heights
    pos_avg = torch.zeros(res_map*res_map, num_avg_heights, 3).to(device) # (L*L, A, 3)
    for i, h in enumerate(np.linspace(height_c-tolerance_c, height_c+tolerance_c, num_avg_heights)):
        pos_avg[:,i,:2] = pos
        pos_avg[:,i,2] = h

    return pos_avg.reshape(-1, 3) # (L*L*A, 3)

def distToCubeBorder(
    rays_o:torch.Tensor,
    rays_d:torch.Tensor,
    cube_min:float,
    cube_max:float,
):
    """
    Calculate distance from ray origins to cube borders.
    Cube is considered to be of equal length in all dimensions.
    Args:
        rays_o: ray origins; tensor of shape (N, 3)
        rays_d: ray directions; tensor of shape (N, 3)
        cube_min: minimum cube coordinate; float
        cube_max: maximum cube coordinate; float
    Returns:
        dists: distance from ray origins to cube borders; tensor of shape (N,)
    """
    dists = torch.inf * torch.ones(rays_o.shape, device=rays_o.device, dtype=torch.float32) # (N, 3)
    dists[rays_d > 0] = (cube_max - rays_o)[rays_d > 0] / rays_d[rays_d > 0]
    dists[rays_d < 0] = (cube_min - rays_o)[rays_d < 0] / rays_d[rays_d < 0]
    return torch.min(dists, dim=1)[0] # (N,)