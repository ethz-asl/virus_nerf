import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import matplotlib.ticker as mtick
import os
import cv2 as cv

from args.args import Args
from modules.rendering import MAX_SAMPLES
from helpers.data_fcts import smoothIgnoreNans
from helpers.plotting_fcts import combineImgs
from training.trainer_base import TrainerBase
from datasets.dataset_base import DatasetBase


class TrainerPlot(TrainerBase):
    def __init__(
        self, 
        hparams_file=None,
        args:Args=None,
        train_dataset:DatasetBase=None,
        test_dataset:DatasetBase=None,
    ):
        TrainerBase.__init__(
            self,
            args=args,
            hparams_file=hparams_file,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

        self.colors = {
            'robot':    'red',
            'GT_map':   'grey', 
            'GT_scan':  'black',
            'NeRF':     'darkorange',
            'LiDAR':    'darkmagenta',
            'USS':      'blue',
            'ToF':      'green',
            'camera':   'lime',
        }

    def _plotOccGrid(
        self,
        step,
    ):
        """
        Plot occupancy grid.
        Args:
            step: current step; int
        """
        if not self.args.eval.plot_results:
            return

        if step % self.grid_update_interval != 0:
            return

        # calculate mean height in cube, world and occupancy grid coordinates
        height_c = self.train_dataset.getMeanHeight()
        height_w = self.train_dataset.scene.c2w(pos=np.array([[0.0, 0.0, height_c]]), copy=False)[0,2]
        height_o = self.model.occupancy_grid.c2oCoordinates(
            pos_c=height_c,
        )

        occ_3d_grid = self.model.occupancy_grid.getOccupancyCartesianGrid(
            clone=True,
        )
        bin_3d_grid = self.model.occupancy_grid.getBinaryCartesianGrid(
            threshold=self.model.occupancy_grid.threshold,
        )

        # verify that the binary grid is correct
        if self.args.training.debug_mode:
            bitfield = self.model.occupancy_grid.getBitfield(
                clone=True,
            )
            bin_morton_grid = self.model.occupancy_grid.bitfield2morton(
                bin_bitfield=bitfield,
            )
            bin_3d_recovery = self.model.occupancy_grid.morton2cartesian(
                grid_morton=bin_morton_grid,
            )

            if not torch.allclose(bin_3d_grid, bin_3d_recovery):
                self.args.logger.error(f"bin_3d_grid and bin_3d_recovery are not the same")

        # convert from 3D to 2D
        occ_2d_grid = occ_3d_grid[:,:,height_o]
        bin_2d_grid = bin_3d_grid[:,:,height_o]

        # convert from tensor to array
        occ_2d_grid = occ_2d_grid.detach().clone().cpu().numpy()
        bin_2d_grid = bin_2d_grid.detach().clone().cpu().numpy()

        # create density maps
        density_map_gt = self.test_dataset.scene.getSliceMap(
            height=height_w, 
            res=occ_2d_grid.shape[0], 
            height_tolerance=self.args.eval.height_tolerance, 
            height_in_world_coord=True
        ) # (L, L)

        # plot occupancy grid
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(9,3))
        axes = axes.flatten()
        scale = self.args.model.scale
        extent = self.test_dataset.scene.c2w(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
        extent = extent.T.flatten()

        ax = axes[0]
        im = ax.imshow(density_map_gt.T, origin='lower', extent=extent, cmap='jet', vmin=0, vmax=1, interpolation='none')
        ax.set_xlabel(f'x [m]')
        ax.set_ylabel(f'y [m]')
        ax.set_title(f'GT')
        if self.args.model.grid_type == 'ngp':
            fig.colorbar(im, ax=ax)

        ax = axes[1]
        if self.args.model.grid_type == 'occ':
            vmax = 1
        else:
            vmax = 10 * (0.01 * MAX_SAMPLES / 3**0.5)
        im = ax.imshow(occ_2d_grid.T, origin='lower', cmap='jet', extent=extent, vmin=0, vmax=vmax, interpolation='none')
        ax.set_xlabel(f'x [m]')
        ax.set_title(f'OccGrid density')
        if self.args.model.grid_type == 'ngp':
            fig.colorbar(im, ax=ax)

        ax = axes[2]
        im = ax.imshow(bin_2d_grid.T, origin='lower', cmap='jet', extent=extent, interpolation='none')
        ax.set_xlabel(f'x [m]')
        ax.set_title(f'OccGrid binary')
        if self.args.model.grid_type == 'ngp':
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Density', rotation=270, labelpad=15)

        # add colorbar
        if self.args.model.grid_type == 'occ':
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.87, 0.1, 0.05, 0.8]) # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)           
            cbar_ax.set_ylabel('Density', rotation=270, labelpad=15)

        # check if directory exists
        if not os.path.exists(os.path.join(self.args.save_dir, "occgrids")):
            os.makedirs(os.path.join(self.args.save_dir, "occgrids"))

        if self.args.model.grid_type == 'ngp':
            plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "occgrids", f"occgrid_{step}.png"))

        plt.show()

    @torch.no_grad()
    def _plotMaps(
            self,
            data_dict:dict,
            metrics_dict:dict,
            num_points:int,
    ):
        """
        Plot scan and density maps.
        Args:
            data_w: data dictionary in world coordinates
            metrics_dict: metrics dictionary
            num_points: number of images to plot
        """
        if not self.args.eval.plot_results:
            return
        
        N = num_points
        N_down = self.args.eval.num_plot_pts

        # save folder
        save_dir = os.path.join(self.args.save_dir, "maps")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # plot
        scale = self.args.model.scale
        extent = self.test_dataset.scene.c2w(pos=np.array([[-scale,-scale],[scale,scale]]), copy=False)
        extent = extent.T.flatten()
        num_ray_steps = 512
        max_error_m = 4.0
        bin_size = 0.2
        hist_bins = np.linspace(0, max_error_m, int(max_error_m/bin_size+1))
        kernel = np.ones((3,3),np.uint8)
        orientation_arrow_length = 0.4
        orientation_arrow_width = 0.001
        pos_circle_radius = 0.06

        for i in np.linspace(0, N-1, N_down, dtype=int):
            fig, axes = plt.subplots(ncols=3, nrows=len(self.args.eval.sensors)-1, figsize=(9,10))

            map_gt = self.test_dataset.scene.getSliceMap(
                height=data_dict['GT']['rays_o'].reshape(N, -1, 3)[i,0,2], 
                res=self.args.eval.res_map, 
                height_tolerance=self.args.eval.height_tolerance, 
                height_in_world_coord=True
            )

            robot_pos = np.array([
                data_dict['robot']['pos']['LiDAR'][i],
                data_dict['robot']['pos']['CAM1'][i],
                data_dict['robot']['pos']['CAM3'][i],
            ])
            robot_orientation = np.array([
                data_dict['robot']['orientation']['LiDAR'][i],
                data_dict['robot']['orientation']['CAM1'][i],
                data_dict['robot']['orientation']['CAM3'][i],
            ])

            for s, sensor in enumerate(self.args.eval.sensors):

                if sensor == 'GT':
                    continue
        
                pos = data_dict[sensor]['pos'].reshape(N, -1, 2)[i] # (M, 2)
                pos_o = data_dict[sensor]['pos_o'].reshape(N, -1, 2)[i] # (M, 2)
                pos_gt = data_dict[sensor]['pos_gt'].reshape(N, -1, 2)[i] # (M, 2)

                scan = self.test_dataset.scene.pos2map(
                    pos=pos,
                    num_points=1,
                ) # (1, L, L)
                scan_gt = self.test_dataset.scene.pos2map(
                    pos=pos_gt,
                    num_points=1,
                ) # (1, L, L)
                scan = cv.dilate(scan[0].astype(np.uint8), kernel, iterations=1) # (L, L)
                scan_gt = cv.dilate(scan_gt[0].astype(np.uint8), kernel, iterations=1) # (L, L)

                img = combineImgs(
                    bool_imgs=[map_gt, scan_gt, scan],
                    colors=[self.colors['GT_map'], self.colors['GT_scan'], self.colors[sensor]],
                    upsample=1,
                ) # (L, L)

                nn_dists = metrics_dict[sensor]['nn_dists'].reshape(N, -1)[i] # (M,)
                nn_dists_inv = metrics_dict[sensor]['nn_dists_inv'].reshape(N, -1)[i] # (M,)
                nn_dists = nn_dists[~np.isnan(nn_dists)]
                nn_dists_inv = nn_dists_inv[~np.isnan(nn_dists_inv)]

                ax = axes[s-1,0]
                ax.imshow(img.swapaxes(0,1), origin='lower', extent=extent, interpolation='none')
                for j in np.linspace(0, pos_o.shape[0]-1, num_ray_steps, dtype=int):
                    xs = [pos_o[j,0], pos[j,0]]
                    ys = [pos_o[j,1], pos[j,1]]
                    ax.plot(xs, ys, c=self.colors[sensor], linewidth=0.1, alpha=0.2)
                for j in range(robot_pos.shape[0]):
                    ax.add_patch(
                        mpatches.Circle((robot_pos[j,0], robot_pos[j,1]), radius=pos_circle_radius, color=self.colors['robot'])
                    )
                    ax.add_patch(
                        mpatches.Arrow(robot_pos[j,0], robot_pos[j,1], orientation_arrow_length*np.cos(robot_orientation[j]), 
                                        orientation_arrow_length*np.sin(robot_orientation[j]), color=self.colors['robot'],
                                        width=orientation_arrow_width)
                    )
                ax.set_xlabel(f'x [m]')
                sensor_label = sensor
                if sensor == "ToF":
                    sensor_label = "IRS"
                ax.set_ylabel(sensor_label, fontsize=15, weight='bold', labelpad=20)
                ax.text(-0.17, 0.5, 'y [m]', fontsize=10, va='center', rotation='vertical', transform=ax.transAxes)

                ax = axes[s-1,1]
                if len(nn_dists_inv) > 0:
                    bin_counts, _, _ = ax.hist(nn_dists, bins=hist_bins, color=self.colors[sensor])
                    ax.vlines(np.mean(nn_dists), ymin=0, ymax=np.max(bin_counts)+1, colors='r', linestyles='dashed', 
                            label=f"Mean: {np.mean(nn_dists):.2f}m")   
                ax.set_ylabel(f'# elements')
                ax.set_xlabel(f'NND [m]')
                if len(ax.get_legend_handles_labels()[1]) > 0:
                    ax.legend()
                ax.set_box_aspect(1)
                ax.set_xlim([0, 1.2*np.max(nn_dists, initial=0.2)])
                ax.set_ylim([0, 1.2*np.max(bin_counts, initial=1.0)])

                ax = axes[s-1,2]
                if len(nn_dists_inv) > 0:
                    bin_counts_inv, _, _ = ax.hist(nn_dists_inv, bins=hist_bins, color=self.colors[sensor])
                    ax.vlines(np.mean(nn_dists_inv), ymin=0, ymax=np.max(bin_counts_inv)+1, colors='r', linestyles='dashed', 
                            label=f"Mean: {np.mean(nn_dists_inv):.2f}m")
                ax.set_ylabel(f'# elements')
                ax.set_xlabel(f'NND [m]')
                if len(ax.get_legend_handles_labels()[1]) > 0:
                    ax.legend()
                ax.set_box_aspect(1)
                ax.set_xlim([0, 1.2*np.max(nn_dists_inv, initial=0.2)])
                ax.set_ylim([0, 1.2*np.max(bin_counts_inv, initial=1.0)])

            axes[0,0].set_title(f'Scan', weight='bold')
            axes[0,1].set_title(f'NND Sensor->GT', weight='bold')
            axes[0,2].set_title(f'NND GT->Sensor', weight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"map{i}.png"))

    def _plotMetrics(
        self,
        metrics_dict:dict,
    ):
        """
        Plot average metrics.
        Args:
            metrics_dict: dict of metrics
        """
        if not self.args.eval.plot_results:
            return

        sensors = list(metrics_dict.keys())
        zones = list(metrics_dict[sensors[0]]['nn_mean'].keys())

        x = np.arange(len(zones))  # the label locations
        width = 0.6  # the width of the bars

        fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(13,8), gridspec_kw={'width_ratios': [5.5, 5.5, 3.5]})
        metrics = [
            'nn_mean', 'nn_mean_inv', 'nn_mean_inv_360', 
            'nn_median', 'nn_median_inv', 'nn_median_inv_360', 
            'nn_inlier', 'nn_inlier_inv', 'nn_inlier_inv_360',
        ]

        y_axis_inv_mean_max = 0.0
        y_axis_inv_median_max = 0.0
        for i, (ax, metric) in enumerate(zip(axs.flatten(), metrics)):

            for j, sensor in enumerate(sensors):

                x_axis = x - width/2 + (j+0.5)*width/len(sensors)
                performances = np.array([metrics_dict[sensors[j]][metric][z] for z in zones])

                if i < 6:
                    if (i%3) != 0:
                        if i < 3:
                            y_axis_inv_mean_max = max(y_axis_inv_mean_max, np.max(performances))
                        else:
                            y_axis_inv_median_max = max(y_axis_inv_median_max, np.max(performances))

                    if (i+1) % 3 == 0:
                        ax.bar(x_axis, performances, width/len(sensors), color=self.colors[sensor])
                    else:
                        sensor_label = sensor
                        if sensor == "ToF":
                            sensor_label = "IRS"
                        ax.bar(x_axis, performances, width/len(sensors), label=sensor_label, color=self.colors[sensor])
                    continue

                nn_outlier_too_close = np.array([metrics_dict[sensors[j]]['nn_outlier_too_close'][z] for z in zones])
                nn_outlier_too_far = 1 - performances - nn_outlier_too_close
                
                if (((i + j) % 2) == 0) and (i < 8):
                    ax.bar(x_axis, performances, width/len(sensors), label='Inliers', color=self.colors[sensor])
                    ax.bar(x_axis, nn_outlier_too_close, width/len(sensors), bottom=performances, 
                            label='Outliers \n(too close)', color=self.colors[sensor], alpha=0.4)
                    ax.bar(x_axis, nn_outlier_too_far, width/len(sensors), bottom=1-nn_outlier_too_far, 
                            label='Outliers \n(too far)', color=self.colors[sensor], alpha=0.1)
                else:
                    ax.bar(x_axis, performances, width/len(sensors), color=self.colors[sensor])
                    ax.bar(x_axis, nn_outlier_too_close, width/len(sensors), bottom=performances, 
                            color=self.colors[sensor], alpha=0.4)
                    ax.bar(x_axis, nn_outlier_too_far, width/len(sensors), bottom=1-nn_outlier_too_far, 
                            color=self.colors[sensor], alpha=0.1)
                    
            if (i+1) % 3 == 0:  
                ax.set_xlim([-0.75*width, np.max(x)+0.75*width])
            else: 
                ax.set_xlim([-0.75*width, np.max(x)+2.75*width])
                ax.legend()

            if i < 6:
                ax.set_xticks(x, [])
            else:
                ax.set_xticks(x, [f"{self.args.eval.zones[z][0]}-{self.args.eval.zones[z][1]}m" for z in zones])
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))        
                
        axs[0,1].set_ylim([0.0, 1.05*y_axis_inv_mean_max])
        axs[0,2].set_ylim([0.0, 1.05*y_axis_inv_mean_max])
        axs[1,1].set_ylim([0.0, 1.05*y_axis_inv_median_max])
        axs[1,2].set_ylim([0.0, 1.05*y_axis_inv_median_max])
        axs[2,0].set_ylim([0.0, 1.05])
        axs[2,1].set_ylim([0.0, 1.05])
        axs[2,2].set_ylim([0.0, 1.05])
        axs[0,0].set_ylabel('Mean [m]')
        axs[1,0].set_ylabel('Median [m]')
        axs[2,0].set_ylabel('Inliers [%]')
        axs[0,0].set_title('Accuracy: Sensor->GT(FoV)') 
        axs[0,1].set_title('Coverage: GT(FoV)->Sensor') 
        axs[0,2].set_title('Coverage: GT(360°)->Sensor') 

        fig.suptitle('Nearest Neighbour Distance', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, f"metrics.png"))

    def _plotLosses(
        self,
        logs:dict,
        metrics_dict:dict,
    ):
        """
        Plot losses, mean-nearest-neighbour distance and peak signal-to-noise-ratio.
        Args:
            logs: logs dictionary
            metrics_dict: dict of metrics
        Returns:
            metrics_dict: dict of metrics
        """
        if (not self.args.eval.plot_results) or (self.args.training.max_steps == 0):
            return metrics_dict
        
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,8))

        # plot losses
        ax = axes[0]
        filter_size = max(self.args.eval.eval_every_n_steps+1, 4)
        ax.plot(logs['step'], smoothIgnoreNans(logs['loss'], filter_size), label='total', c="black")
        ax.plot(logs['step'], smoothIgnoreNans(logs['color_loss'], filter_size), label='color', c=self.colors['camera'])
        if "rgbd_loss" in logs:
            ax.plot(logs['step'], smoothIgnoreNans(logs['rgbd_loss'], filter_size), label='rgbd')
        if "ToF_loss" in logs:
            ax.plot(logs['step'], smoothIgnoreNans(logs['ToF_loss'], filter_size), label='IRS', c=self.colors['ToF'])
        if "USS_loss" in logs:
            ax.plot(logs['step'], smoothIgnoreNans(logs['USS_loss'], filter_size), label='USS', c=self.colors['USS'])
        ax.set_ylabel('loss')
        ax.set_ylim([0, 1.0])

        ax.set_xlabel('step')
        secax = ax.secondary_xaxis(
            location='top', 
            functions=(self._step2time, self._time2step),
        )
        secax.set_xlabel('time [s]')
        ax.legend()
        ax.set_title('Losses')

        # plot mnn and psnr 
        if 'mnn' in logs and 'psnr' in logs:
            ax = axes[1]
            color = self.colors['NeRF']
            not_nan = ~np.isnan(logs['mnn'])
            lns1 = ax.plot(np.array(logs['step'])[not_nan], np.array(logs['mnn'])[not_nan], c=color, label='mnn')
            hln1 = ax.axhline(metrics_dict['NeRF']['nn_mean']['zone3'], linestyle="--", c=color, label='mnn final')
            ax.set_ylabel('Mean NND [m]')
            ax.set_ylim([0, 0.5])
            ax.yaxis.label.set_color('blue') 
            ax.tick_params(axis='y', colors='blue')

            ax2 = ax.twinx()
            color = self.colors['camera']
            not_nan = ~np.isnan(logs['psnr'])
            lns2 = ax2.plot(np.array(logs['step'])[not_nan], np.array(logs['psnr'])[not_nan], label='psnr', c=color)
            # ax.axhline(metrics_dict['NeRF']['psnr'], linestyle="--", c=color, label='psnr final')
            ax2.set_ylabel('PSNR')
            ax2.yaxis.label.set_color('green') 
            ax2.tick_params(axis='y', colors='green')

            ax.set_xlabel('step')
            secax = ax.secondary_xaxis(
                location='top', 
                functions=(self._step2time, self._time2step),
            )
            secax.set_xlabel('time [s]')
            lns = lns1 + lns2 + [hln1]
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_title('Metrics')

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "losses.png"))

        return metrics_dict
    
