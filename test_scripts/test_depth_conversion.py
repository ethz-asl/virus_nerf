import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from alive_progress import alive_bar
 
sys.path.insert(0, os.getcwd())
from datasets.ray_utils import get_rays
from datasets.RH2 import RobotAtHomeDataset
from training.trainer_rh import TrainerRH
from training.metrics_rh import MetricsRH
from args.args import Args

def createScanMap(
            dataset:RobotAtHomeDataset,
            rays_o_w:np.array,
            depth:np.array,
            scan_angles:np.array,
            map_res:int,
    ):
        """
        Create scan map for given rays and depths.
        Args:
            dataset: RobotAtHomeDataset object
            rays_o_w: ray origins in world coordinates (meters); numpy array of shape (N*M, 3)
            depth: depths in wolrd coordinates (meters); numpy array of shape (N*M,)
            scan_angles: scan angles; numpy array of shape (N*M,)
            map_res: map resolution; int
        Returns:
            scan_map: scan maps; numpy array of shape (map_res, map_res)
        """
        # convert depth to position in world coordinate system and then to map indices
        pos = dataset.scene.depth2pos(rays_o=rays_o_w, scan_depth=depth, scan_angles=scan_angles) # (N*M, 2)
        idxs = dataset.scene.w2idx(pos=pos, res=map_res) # (N*M, 2)

        # create scan map
        scan_map = np.zeros((map_res, map_res))
        scan_map[idxs[:,0], idxs[:,1]] = 1.0

        return scan_map # (map_res, map_res)


def main():
    # set hyperparameters
    res_map = 256
    height_tol = 0.01
    num_imgs = 10
    num_pixs = 100
    alphas = np.linspace(0.98, 1.02, 41)
    betas = np.linspace(0, 0.01, 41)

    # load hparams
    hparams_file = "rh_anto_kitchen_win.json" # "rh_windows.json"
    args = Args(file_name=hparams_file)
    args.rh.sensor_model = "RGBD"
    args.dataset.keep_N_observations = "all"
    args.rh.room = "kitchen1"

    # load dataset
    dataset = RobotAtHomeDataset(
        args = args,
        split='test',
    ).to(args.device)

    # metric class
    W, H = dataset.img_wh
    metrics = MetricsRH(
        args=args,
        rh_scene=dataset.scene,
        img_wh=(W,H)
    )

    # create image and pixel indices
    img_idxs = np.linspace(0, len(dataset)-1, num_imgs, dtype=int) # (num_imgs,)
    img_idxs = np.repeat(img_idxs, num_pixs) # (num_imgs*num_pixs,)

    pix_idxs = np.arange(W*H).reshape(H, W) # (H, W)
    pix_idxs = pix_idxs[H//2,np.linspace(0, W-1, num_pixs, dtype=int)] # (num_pixs,)
    pix_idxs = np.tile(pix_idxs, num_imgs) # (num_imgs*num_pixs,)
    
    # get positions and directions
    direction = dataset.directions[pix_idxs]
    pose = dataset.poses[img_idxs]
    rays_o, rays_d = get_rays(direction, pose)

    scan_map_gt, depth_c_gt, scan_angles = dataset.scene.getSliceScan(
        res=res_map, 
        rays_o=rays_o.detach().cpu().numpy(), 
        rays_d=rays_d.detach().cpu().numpy(), 
        rays_o_in_world_coord=False, 
        height_tolerance=height_tol
    ) # (L, L)
    depth_w_gt = dataset.scene.c2w(pos=depth_c_gt, only_scale=True, copy=True)

    rays_o_w = dataset.scene.c2w(pos=rays_o.detach().cpu().numpy(), copy=True)
    rmses = np.zeros((len(alphas), len(betas)))
    maes = np.zeros((len(alphas), len(betas)))
    nns = np.zeros((len(alphas), len(betas)))
    with alive_bar(len(alphas), bar = 'bubbles', receipt=False) as bar:
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # get depths and convert it to world coordinates
                depth_c = dataset.depths[img_idxs, pix_idxs] # (num_imgs*num_pixs,)
                depth_c = beta + alpha*depth_c.clone().detach().cpu().numpy()
                depth_w = dataset.scene.c2w(pos=depth_c, only_scale=True, copy=True)

                # calculate nearest neighbor error
                metric_dict = metrics.evaluate(
                    data={
                        "depth": depth_w,
                        "depth_gt": depth_w_gt,
                        "rays_o": rays_o_w,
                        "scan_angles": scan_angles,
                    },
                    eval_metrics=["rmse", "mae", "nn"],
                    convert_to_world_coords=False,
                    copy=True,
                    num_test_pts=num_imgs,
                )
                rmses[i,j] = metric_dict["rmse"]
                maes[i,j] = metric_dict["mae"]
                nns[i,j] = metric_dict["mnn"]

                if alpha==alphas[len(alphas)//2] and beta==betas[len(betas)//2]:
                    depth_w_zero = np.copy(depth_w)
                    scan_map_zero = createScanMap(
                        dataset=dataset,
                        rays_o_w=rays_o_w,
                        depth=depth_w,
                        scan_angles=scan_angles,
                        map_res=res_map,
                    ) # (map_res, map_res)
            bar()

    

    # print best alpha and beta
    best_idx = np.argmin(rmses)
    best_alpha_idx = best_idx // len(betas)
    best_beta_idx = best_idx % len(betas)
    best_pair = (alphas[best_alpha_idx], betas[best_beta_idx])
    best_beta_c = dataset.scene.c2w(pos=best_pair[1], only_scale=True, copy=True)
    print(f"\nRMSE:")
    print(f"best pair (alpha,beta): {best_pair}")
    print(f"best beta in world coordinates: {best_beta_c}")
    print(f"best alpha for beta=0: {alphas[np.argmin(rmses[:,0])]}")

    best_idx = np.argmin(maes)
    best_alpha_idx = best_idx // len(betas)
    best_beta_idx = best_idx % len(betas)
    best_pair = (alphas[best_alpha_idx], betas[best_beta_idx])
    best_beta_c = dataset.scene.c2w(pos=best_pair[1], only_scale=True, copy=True)
    print(f"\nMARE:")
    print(f"best pair (alpha,beta): {best_pair}")
    print(f"best beta in world coordinates: {best_beta_c}")
    print(f"best alpha for beta=0: {alphas[np.argmin(maes[:,0])]}")

    best_idx = np.argmin(nns)
    best_alpha_idx = best_idx // len(betas)
    best_beta_idx = best_idx % len(betas)
    best_pair = (alphas[best_alpha_idx], betas[best_beta_idx])
    best_beta_c = dataset.scene.c2w(pos=best_pair[1], only_scale=True, copy=True)
    print(f"\nNN:")
    print(f"best pair (alpha,beta): {best_pair}")
    print(f"best beta in world coordinates: {best_beta_c}")
    print(f"best alpha for beta=0: {alphas[np.argmin(nns[:,0])]}")

    # plot
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12,8))

    scale = args.model.scale
    extent = dataset.scene.c2w(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
    extent = extent.T.flatten()
    pos_robot = np.unique(rays_o_w, axis=0)
    pos_obj = rays_o_w + np.stack([depth_w_zero*np.cos(scan_angles), depth_w_zero*np.sin(scan_angles), np.zeros_like(depth_w_zero)], axis=1)
    pos_obj_gt = rays_o_w + np.stack([depth_w_gt*np.cos(scan_angles), depth_w_gt*np.sin(scan_angles), np.zeros_like(depth_w_gt)], axis=1)

    ax = axes[0,0]
    ax.imshow(scan_map_gt.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(scan_map_gt))
    ax.set_title(f'Scan GT', weight='bold')
    ax.set_xlabel(f'x [m]')
    ax.set_ylabel(f'y [m]')
    ax.scatter(pos_robot[:,0], pos_robot[:,1], color='w', s=1)
    for i in range(rays_o_w.shape[0]):
        ax.plot([rays_o_w[i,0], pos_obj_gt[i,0]], [rays_o_w[i,1], pos_obj_gt[i,1]], c='w', linewidth=0.1)

    ax = axes[0,1]
    ax.imshow(scan_map_zero.T, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=np.max(scan_map_zero))
    ax.set_title(f'Scan', weight='bold')
    ax.set_xlabel(f'x [m]')
    ax.set_ylabel(f'y [m]')
    ax.scatter(pos_robot[:,0], pos_robot[:,1], color='w', s=1)
    for i in range(rays_o_w.shape[0]):
        ax.plot([rays_o_w[i,0], pos_obj[i,0]], [rays_o_w[i,1], pos_obj[i,1]], c='w', linewidth=0.1)

    step = np.max((len(betas)//3, len(alphas)//3))
    if step==0:
        step = 1
    
    ax = axes[0,2]
    im0 = ax.imshow(rmses, origin='lower',  cmap='viridis', vmin=np.min(rmses), vmax=np.max(rmses))
    ax.set_ylabel(f'alpha')
    ax.set_xlabel(f'beta')
    ax.set_xticks(np.arange(len(betas))[::step], np.round(betas[::step], 4))
    ax.set_yticks(np.arange(len(alphas))[::step], np.round(alphas[::step], 4))
    ax.set_title(f'RMSE', weight='bold')

    ax = axes[1,2]
    im1 = ax.imshow(maes, origin='lower',  cmap='viridis', vmin=np.min(maes), vmax=np.max(maes))
    ax.set_ylabel(f'alpha')
    ax.set_xlabel(f'beta')
    ax.set_xticks(np.arange(len(betas))[::step], np.round(betas[::step], 4))
    ax.set_yticks(np.arange(len(alphas))[::step], np.round(alphas[::step], 4))
    ax.set_title(f'MAE', weight='bold')

    ax = axes[2,2]
    im2 = ax.imshow(nns, origin='lower',  cmap='viridis', vmin=np.min(nns), vmax=np.max(nns))
    ax.set_ylabel(f'alpha')
    ax.set_xlabel(f'beta')
    ax.set_xticks(np.arange(len(betas))[::step], np.round(betas[::step], 4))
    ax.set_yticks(np.arange(len(alphas))[::step], np.round(alphas[::step], 4))
    ax.set_title(f'NN', weight='bold')

    ax = axes[1,0]
    rgb = dataset.rays[img_idxs[0]][:, :3]
    rgb = rgb.reshape(H, W, 3).detach().cpu().numpy()
    ax.imshow(rgb)
    ax.set_title(f'RGB 0', weight='bold')

    ax = axes[1,1]
    rgb = dataset.rays[img_idxs[-1]][:, :3]
    rgb = rgb.reshape(H, W, 3).detach().cpu().numpy()
    ax.imshow(rgb)
    ax.set_title(f'RGB -1', weight='bold')

    ax = axes[2,0]
    rgbd = dataset.depths[img_idxs[0]]
    rgbd = rgbd.reshape(H, W).detach().cpu().numpy()
    ax.imshow(rgbd)
    ax.set_title(f'Depth 0', weight='bold')

    ax = axes[2,1]
    rgbd = dataset.depths[img_idxs[-1]]
    rgbd = rgbd.reshape(H, W).detach().cpu().numpy()
    ax.imshow(rgbd)
    ax.set_title(f'Depth -1', weight='bold')

    cbar = plt.colorbar(im0)
    cbar = plt.colorbar(im1)
    cbar = plt.colorbar(im2)
    # cbar.set_label('Color Bar Label')
    plt.tight_layout()
    plt.show()
   
if __name__ == "__main__":
    main()