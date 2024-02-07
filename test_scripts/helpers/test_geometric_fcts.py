import torch
import os
import sys
 
sys.path.insert(0, os.getcwd())
from helpers.geometric_fcts import distToCubeBorder


def test_distToCubeBorder():
    rays_o = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rays_d = torch.tensor([[0.0, 0.0, 2.0], [0.0, 1.5, -1.0]])
    cube_min = -0.5
    cube_max = 0.5
    dist = distToCubeBorder(rays_o, rays_d, cube_min, cube_max)

    print(dist)


if __name__ == "__main__":
    test_distToCubeBorder()