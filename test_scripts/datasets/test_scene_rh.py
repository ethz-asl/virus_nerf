import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from robotathome import RobotAtHome
 
sys.path.insert(0, os.getcwd())
from datasets.scene_rh import SceneRH


def test_RobotAtHomeScene():
    # load dataset
    my_rh_path = '../RobotAtHome2/data'
    my_rgbd_path = os.path.join(my_rh_path, 'files/rgbd')
    my_scene_path = os.path.join(my_rh_path, 'files/scene')
    my_wspc_path = 'results'
    my_db_filename = "rh.db"
    rh = RobotAtHome(
        rh_path=my_rh_path, 
        rgbd_path=my_rgbd_path, 
        scene_path=my_scene_path, 
        wspc_path=my_wspc_path, 
        db_filename=my_db_filename
    )

    # load scene
    rh_location_names = {
        "session": "session_2",
        "home": "anto",
        "room": "livingroom1",
        "subsession": "subsession_1",
        "home_session": "s1",
    }
    rh_scene = SceneRH(rh, rh_location_names)

    # get slice map
    res = 256
    res_angular = 256
    rays_o_w = np.array([[0,0,0.5]])
    rays_o_w = np.repeat(rays_o_w, res_angular, axis=0)
    slice_map = rh_scene.getSliceMap(height=rays_o_w[0,2], res=res)

    # get slice scan
    rays_d_is_given = True
    if rays_d_is_given:
        rays_o_w2 = np.array([[0,-2,0.5]])
        rays_o_w2 = np.repeat(rays_o_w2, res_angular, axis=0)
        angles_d1 = np.linspace(-np.pi/3, np.pi/3, res_angular, endpoint=False)
        rays_d1 = np.stack((np.cos(angles_d1), np.sin(angles_d1), np.zeros_like(angles_d1)), axis=1)
        angles_d2 = np.linspace(0, 2*np.pi/3, res_angular, endpoint=False)
        rays_d2 = np.stack((np.cos(angles_d2), np.sin(angles_d2), np.zeros_like(angles_d2)), axis=1)

        rays_o_w = np.concatenate((rays_o_w, rays_o_w2), axis=0)
        rays_d = np.concatenate((rays_d1, rays_d2), axis=0)

        scan_map, scan_depth_c, scan_angles = rh_scene.getSliceScan(res=res, rays_o=rays_o_w, rays_d=rays_d, rays_o_in_world_coord=True)
    else:
        scan_map, scan_depth_c, scan_angles = rh_scene.getSliceScan(res=res, rays_o=rays_o_w, rays_d=None, rays_o_in_world_coord=True)

    # convert scan depth to position in world coordinate system
    rays_o_c = rh_scene.w2c(pos=rays_o_w, copy=True)
    scan_depth_pos_c = rh_scene.depth2pos(rays_o=rays_o_c, scan_depth=scan_depth_c, scan_angles=scan_angles) # (N, 2)
    scan_depth_pos_w = rh_scene.c2w(pos=scan_depth_pos_c, copy=True) # (N, 2)

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,6))
    extent = rh_scene.c2w(pos=np.array([[-0.5,-0.5],[0.5,0.5]]), copy=False)
    extent = extent.T.flatten()
    rays_o_w_unique = np.unique(rays_o_w, axis=0)
    comb_map = slice_map + 2*scan_map
    # score = np.sum(slice_map * slice_scans[i]) / np.sum(slice_scans[i])

    ax = axes[0]
    ax.imshow(slice_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    ax.scatter(rays_o_w_unique[:,0], rays_o_w_unique[:,1], c='w', s=5)
    ax.set_title(f'Map')
    ax.set_xlabel(f'x [m]')
    ax.set_ylabel(f'y [m]')

    ax = axes[1]
    ax.imshow(2*scan_map.T,origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    for i in range(scan_depth_pos_w.shape[0]):
        ax.plot([rays_o_w[i,0], scan_depth_pos_w[i,0]], [rays_o_w[i,1], scan_depth_pos_w[i,1]], c='w', linewidth=0.1)
    ax.scatter(rays_o_w_unique[:,0], rays_o_w_unique[:,1], c='w', s=5)
    ax.set_title(f'Scan')
    ax.set_xlabel(f'x [m]')
    
    ax = axes[2]
    ax.imshow(comb_map.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(comb_map))
    for i in range(scan_depth_pos_w.shape[0]):
        ax.plot([rays_o_w[i,0], scan_depth_pos_w[i,0]], [rays_o_w[i,1], scan_depth_pos_w[i,1]], c='w', linewidth=0.1)
    ax.scatter(rays_o_w_unique[:,0], rays_o_w_unique[:,1], c='w', s=5)
    ax.set_title(f'Combined')
    ax.set_xlabel(f'x [m]')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_RobotAtHomeScene()