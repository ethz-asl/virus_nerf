import numpy as np
import torch
from einops import rearrange
from kornia import create_meshgrid


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H,
                       W,
                       K,
                       device='cpu',
                       random=False,
                       return_uv=False,
                       flatten=True):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else:  # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    return directions

@torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(directions, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (N, 3), the origin of the rays in world coordinate
        rays_d: (N, 3), the direction of the rays in world coordinate
    """
    if c2w.ndim == 2:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T
    else:
        rays_d = rearrange(directions, 'n c -> n 1 c') @ \
                 rearrange(c2w[..., :3], 'n a b -> n b a')
        rays_d = rearrange(rays_d, 'n 1 c -> n c')

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., 3].expand_as(rays_d)

    if(rays_d.shape[1] == 4):
        rays_d = rays_d[:, :3]
        rays_o = rays_o[:, :3]

    return rays_o, rays_d

