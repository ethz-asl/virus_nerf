import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
 
sys.path.insert(0, os.getcwd())
from args.args import Args
from modules.occupancy_grid import OccupancyGrid

def plot_occ_grid(
    occ_grid_2d:np.array,
    extent:np.array,
):
    # plot occupancy grid
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

    ax = axes
    im = ax.imshow(occ_grid_2d.T, origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=1)
    ax.set_xlabel(f'x [m]')
    ax.set_ylabel(f'y [m]')
    ax.set_title(f'Occupancy Grid')
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_model_probs(
    dists,
    probs_occ,
    probs_emp,
    probs_equal_occ,
    probs_equal_emp,
    probs_notless_occ,
    probs_notless_emp,
):
    x_min = dists.min()
    x_max = dists.max()
    y_min = np.min((probs_occ.min(), probs_emp.min(), 
                       probs_equal_occ.min(), probs_equal_emp.min(), 
                       probs_notless_occ.min(), probs_notless_emp.min()))
    y_max = np.max((probs_occ.max(), probs_emp.max(),
                          probs_equal_occ.max(), probs_equal_emp.max(),
                          probs_notless_occ.max(), probs_notless_emp.max()))

    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12,8))

    ax = axes[0,0]
    ax.plot(dists[0], probs_occ[0], label="occ")
    ax.plot(dists[0], probs_equal_occ[0], "--", label="probs_equal_occ")
    ax.plot(dists[0], probs_notless_occ[0], "--", label="probs_notless_occ")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ax = axes[1,0]
    ax.plot(dists[1], probs_occ[1], label="occ")
    ax.plot(dists[1], probs_equal_occ[1], "--", label="probs_equal_occ")
    ax.plot(dists[1], probs_notless_occ[1], "--", label="probs_notless_occ")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ax = axes[2,0]
    ax.plot(dists[2], probs_occ[2], label="occ")
    ax.plot(dists[2], probs_equal_occ[2], "--", label="probs_equal_occ")
    ax.plot(dists[2], probs_notless_occ[2], "--", label="probs_notless_occ")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ax = axes[0,1]
    ax.plot(dists[0], probs_emp[0], label="emp")
    ax.plot(dists[0], probs_equal_emp[0], "--", label="probs_equal_emp")
    ax.plot(dists[0], probs_notless_emp[0], "--", label="probs_notless_emp")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ax = axes[1,1]
    ax.plot(dists[1], probs_emp[1], label="emp")
    ax.plot(dists[1], probs_equal_emp[1], "--", label="probs_equal_emp")
    ax.plot(dists[1], probs_notless_emp[1], "--", label="probs_notless_emp")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ax = axes[2,1]
    ax.plot(dists[2], probs_emp[2], label="emp")
    ax.plot(dists[2], probs_equal_emp[2], "--", label="probs_equal_emp")
    ax.plot(dists[2], probs_notless_emp[2], "--", label="probs_notless_emp")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    y_min = -3
    y_max = 3

    ax = axes[0,2]
    ax.plot(dists[0], np.log(probs_occ[0] / probs_emp[0]), label="log lik")
    ax.plot(dists[0], np.log(probs_equal_occ[0] / probs_equal_emp[0]), "--", label="log lik equal")
    ax.plot(dists[0], np.log(probs_notless_occ[0] / probs_notless_emp[0]), "--", label="log lik. notless")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ax = axes[1,2]
    ax.plot(dists[1], np.log(probs_occ[1] / probs_emp[1]), label="log lik.")
    ax.plot(dists[1], np.log(probs_equal_occ[1] / probs_equal_emp[1]), "--", label="log lik. equal")
    ax.plot(dists[1], np.log(probs_notless_occ[1] / probs_notless_emp[1]), "--", label="log lik. notless")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    ax = axes[2,2]
    ax.plot(dists[2], np.log(probs_occ[2] / probs_emp[2]), label="log lik.")
    ax.plot(dists[2], np.log(probs_equal_occ[2] / probs_equal_emp[2]), "--", label="log lik. equal")
    ax.plot(dists[2], np.log(probs_notless_occ[2] / probs_notless_emp[2]), "--", label="log lik. notless")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend()

    plt.show()


def main():
    # load hparams
    hparams_file = "rh_windows.json"
    args = Args(file_name=hparams_file)
    args.model.scale = 6.0

    # create occupancy grid
    occ_grid = OccupancyGrid(
        args=args,
        grid_size=32,
    )

    # update grid
    rays_o = torch.tensor([[0,0,0],
                           [0,1,0],
                           [0,2,0]], dtype=torch.float32)
    rays_d = torch.tensor([[1,0,0],
                           [1,0,0],
                           [1,0,0]], dtype=torch.float32)
    meas = torch.tensor([1, 2, 5], dtype=torch.float32)
    occ_grid.rayUpdate(
        rays_o=rays_o,
        rays_d=rays_d,
        meas=meas,
    )

    # get grid
    grid_3d = occ_grid.grid.detach().cpu().numpy()
    grid_2d = grid_3d[:,:,grid_3d.shape[2]//2]

    print(f"grid_3d min: {grid_3d.min()} max: {grid_3d.max()}")
    print(f"grid_2d min: {grid_2d.min()} max: {grid_2d.max()}")

    # plot grid
    scale = args.model.scale
    extent = np.array([[-scale,-scale],[scale,scale]])
    extent = extent.T.flatten()

    plot_occ_grid(
        occ_grid_2d=grid_2d,
        extent=extent,
    )

    # calculate cell distances
    M = 32
    stds = occ_grid._calcStd(
        dists=meas,
    ) # (N,)
    steps = torch.linspace(0, 1, M, device=args.device, dtype=torch.float32) # (M,)
    cell_dists = steps[None,:] * (meas + 3*stds)[:,None] # (N, M)

    # calculate cell probabilities
    probs_occ, probs_emp, probs_equal_emp, probs_equal_occ, probs_notless_emp, probs_notless_occ = occ_grid._rayMeasProb(
        meas=meas, 
        dists=cell_dists,
        return_probs=True,
    )  # (N, M), (N, M), (N, M), (N, M), (N, M), (N, M)

    plot_model_probs(
        dists=cell_dists.detach().cpu().numpy(),
        probs_occ=probs_occ.detach().cpu().numpy(),
        probs_emp=probs_emp.detach().cpu().numpy(),
        probs_equal_occ=probs_equal_occ.detach().cpu().numpy(),
        probs_equal_emp=probs_equal_emp.detach().cpu().numpy(),
        probs_notless_occ=probs_notless_occ.detach().cpu().numpy(),
        probs_notless_emp=probs_notless_emp.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    main()