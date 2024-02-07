import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import json


class PlotterEthz():
    def __init__(
            self,
            data_dir,
        ) -> None:
        self.data_dir = data_dir

        self.score_min = 0.15
        self.score_max = 0.24
        self.num_particles = 32
        self.keep_best_n_particles = 5
        self.converged_since_n_iters = 10
        self.best_symbs = ['*', 'o', 'd', 'x', '+', 'v', '<', '>', 'p', 'P', 'h', 'H', 'X', 'D', 's', '^', '_']
        self.best_symbs = self.best_symbs[:self.keep_best_n_particles]

    def plot(
            self,
        ):
        """
        Plot results from data_dir.
        """
        # read data
        pos, scores, parameters = self._readPosData()
        vel, parameters_vel = self._readPosData(
            read_vel=True
        )
        best_pos, best_scores, best_iters, best_parameters = self._readBestPosData()
        hparams_lims = self._readHparamsLims()

        # verify that parameters are the same
        assert parameters == best_parameters
        assert parameters == parameters_vel

        # print best score and best hparams
        _, best_particles = self._keepBestNParticles(
            scores=scores,
            best_scores=best_scores,
            arr_list=[],
        )
        best_particle = best_particles[0]
        if np.isnan(pos[best_particle, -1, 0]):
            pos_best = pos[best_particle, -2, :]
        else:
            pos_best = pos[best_particle, -1, :]
        for i, param in parameters.items():
            print(f"{param}: {pos_best[i]}")
        print(f"\nBest particle: {best_particle}, best score: {best_scores[best_particle]}")

        # calculate maximal variation over last TT iterations
        pos_norm = np.zeros_like(pos) # (N, T, M)
        for i, param in parameters.items():
            pos_norm[:, :, i] = (pos[:, :, i] - hparams_lims[param][0]) / (hparams_lims[param][1] - hparams_lims[param][0])
        pos_norm = pos_norm[best_particles, -self.converged_since_n_iters:, :] # (N, 10, M)
        pos_norm = np.linalg.norm(pos_norm, axis=2) # (N, 10)
        pos_var_max = np.nanmax(pos_norm, axis=1) - np.nanmin(pos_norm, axis=1) # (N,)
        print(f"Maximal variation over last {self.converged_since_n_iters} iterations mean: {np.nanmean(pos_var_max)}, "
              +f"min: {np.nanmin(pos_var_max)}, max: {np.nanmax(pos_var_max)}")

        # adjust minimal score
        if np.min(scores) < self.score_min:
            self.score_min = np.min(scores)

        # reverse colotmap
        cmap = matplotlib.colormaps['jet']
        cmap_inv = cmap.reversed() 

        # plot
        fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(14,10))

        axes[0] = self._plotParticleSpeeds(
            vel=vel,
            scores=scores,
            best_iters=best_iters,
            best_scores=best_scores,
            ax=axes[0],
            cmap_inv=cmap_inv,
        )

        axes[1] = self._plotParticleScores(
            scores=scores,
            ax=axes[1],
            cmap_inv=cmap_inv,
        )

        axes[2], im = self._plotHparams(
            pos=pos,
            scores=scores,
            best_pos=best_pos,
            best_scores=best_scores,
            best_iters=best_iters,
            hparams_lims=hparams_lims,
            parameters=parameters,
            ax=axes[2],
            cmap_inv=cmap_inv,
        )

        # add colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.05, 0.8]) # [left, bottom, width, height]
        # cbar_ax.set_title('Mean NND',fontsize=13)
        fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_ylabel('NND [m]', rotation=270, labelpad=15)

        # save figure
        fig.savefig(os.path.join(self.data_dir, 'pso_results.png'))
        plt.show()

    def _plotParticleSpeeds(
        self,
        vel:np.ndarray,
        scores:np.ndarray,
        best_scores:np.ndarray,
        best_iters:np.ndarray,
        ax:matplotlib.axes.Axes,
        cmap_inv:matplotlib.colors.LinearSegmentedColormap,
    ):
        """
        Plot particle speeds.
        Args:
            vel: particle velocities; numpy array of shape (N, T, M)
            scores: particle scores; numpy array of shape (N, T)
            best_scores: best scores; numpy array of shape (N,)
            best_iters: best iterations; numpy array of shape (N,)
            ax: matplotlib axis
            cmap_inv: reversed colormap
        Returns:
            ax: matplotlib axis
        """
        vel_norm = np.linalg.norm(vel, axis=2) # (N, T)
        vel_norm_mean = np.nanmean(vel_norm, axis=0) # (T,)
        vel_norm_std = np.nanstd(vel_norm, axis=0) # (T,)

        ax.plot(np.arange(vel.shape[1]), vel_norm_mean, c='k', label='Mean Speed')
        ax.fill_between(np.arange(vel.shape[1]), vel_norm_mean - vel_norm_std, vel_norm_mean + vel_norm_std, alpha=0.2, color='k', label='Std Speed')

        # keep only best N particles
        best_n_arr_list, best_particles = self._keepBestNParticles(
            scores=scores,
            best_scores=best_scores,
            arr_list=[vel, scores, best_scores, best_iters],
        )
        vel, scores, best_scores, best_iters = best_n_arr_list

        for i in np.arange(vel.shape[0])[::-1]:
            score = scores[i,-self.converged_since_n_iters:]
            ax.scatter(np.arange(vel.shape[1]), vel_norm[i], c=scores[i], 
                           cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[i])
            ax.scatter(np.arange(vel.shape[1])[-2], vel_norm[i][-2], c=scores[i][-2], 
                           cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[i],
                            label=f'Particle {best_particles[i]}, NND: {np.nanmean(score):.3f}')
            
            if self.converged_since_n_iters <= 0:
                ax.scatter(np.arange(vel.shape[1])[best_iters[i]], vel_norm[i, best_iters[i]], c=best_scores[i], 
                            cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[i], s=200,
                            label=f'Particle {best_particles[i]}, best NND: {best_scores[i]:.3f}')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Normalized Speed')
        ax.set_ylim([0, np.nanmax(vel_norm)])
        ax.xaxis.set_label_coords(0.5, -0.09)
        ax.legend(loc='upper right')

        return ax
    
    def _plotParticleScores(
        self,
        scores:np.ndarray,
        ax:matplotlib.axes.Axes,
        cmap_inv:matplotlib.colors.LinearSegmentedColormap,
    ):
        """
        Plot particle scores.
        Args:
            scores: particle scores; numpy array of shape (N, T)
            ax: matplotlib axis
            cmap_inv: reversed colormap
        Returns:
            ax: matplotlib axis
        """

        for i in range(scores.shape[0]):
            score = scores[i,-self.converged_since_n_iters:]
            score = score[~np.isnan(score)]

            c = cmap_inv((np.mean(score) - self.score_min) / (self.score_max - self.score_min))
            ax.boxplot(score, positions=[i], widths=0.7, showfliers=True, whis=[10,90],
                        patch_artist=True, boxprops=dict(facecolor=c, color=c), medianprops=dict(color="black"))

        ax.set_xlabel('Particle')
        ax.set_ylabel('Converged NND [m]')

        return ax
        
    def _plotHparams(
        self,
        pos:np.ndarray,
        scores:np.ndarray,
        best_pos:np.ndarray,
        best_scores:np.ndarray,
        best_iters:np.ndarray,
        hparams_lims:dict,
        parameters:dict,
        ax:matplotlib.axes.Axes,
        cmap_inv:matplotlib.colors.LinearSegmentedColormap,
    ):
        """
        Plot particle scores.
        Args:
            pos: particle positions; numpy array of shape (N, T, M)
            scores: particle scores; numpy array of shape (N, T)
            best_pos: best particle positions; numpy array of shape (N, M)
            best_scores: best scores; numpy array of shape (N,)
            best_iters: best iterations; numpy array of shape (N,)
            hparams_lims: dictionary of hparams limits { hparam_name: [min, max] }; dictionary { str: [float, float] }
            parameters: dictionary of parameters { column_index: parameter_name}; dictionary { int: str }
            ax: matplotlib axis
            cmap_inv: reversed colormap
        Returns:
            ax: matplotlib axis
            im: scatter plot
        """
        column_width = 0.6
        plot_every_n_iters = 10
        N = pos.shape[0]
        T = pos.shape[1]

        # normalize positions
        for i, param in parameters.items():
            pos[:, :, i] = (pos[:, :, i] - hparams_lims[param][0]) / (hparams_lims[param][1] - hparams_lims[param][0])
            best_pos[:, i] = (best_pos[:, i] - hparams_lims[param][0]) / (hparams_lims[param][1] - hparams_lims[param][0])

        # keep only best N particles
        best_n_arr_list, _ = self._keepBestNParticles(
            scores=scores,
            best_scores=best_scores,
            arr_list=[pos, scores, best_pos, best_scores, best_iters],
        )
        pos, scores, best_pos, best_scores, best_iters = best_n_arr_list

        for i, param in parameters.items():

            # x_axis = i + column_width * np.linspace(-0.5, 0.5, T) # (T,)
            for j in np.arange(best_pos.shape[0])[::-1]:

                # start = j * (plot_every_n_iters//best_pos.shape[0])
                # plot_iters = np.arange(start, T, plot_every_n_iters)

                if np.isnan(pos[j, -1, i]).any():
                    plot_iters = -2
                else:
                    plot_iters = -1
                im = ax.scatter(i, pos[j, plot_iters, i].flatten(), c=scores[j,plot_iters], 
                                cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[j])
                
                if self.converged_since_n_iters <= 0:
                    ax.scatter(x_axis[best_iters[j]], best_pos[j, i], c=best_scores[j], 
                            cmap=cmap_inv, vmin=self.score_min, vmax=self.score_max, marker=self.best_symbs[j], s=200)

        ax.set_xticks(list(parameters.keys()))
        ax.set_xticklabels([param.replace('_', ' ').replace(' every m', '') + f":\n     [{hparams_lims[param][0]:.1f}, {hparams_lims[param][1]:.1f}]" 
                            for param in parameters.values()], rotation=30, fontsize=9)
        ax.set_ylabel('Normalized Final Position')
        return ax, im
    
    def _keepBestNParticles(
        self,
        scores:np.ndarray,
        best_scores:np.ndarray,
        arr_list:list
    ):
        """
        Keep only best N particles.
        Args:
            scores: particle scores; numpy array of shape (N, T)
            best_scores: best scores; numpy array of shape (N,)
            arr_list: list of arrays to keep best N particles; list of numpy arrays
        Returns:
            best_n_arr_list: list of best N particles; list of numpy arrays
            best_particles: best particles; numpy array of shape (N,)
        """
        scores = np.copy(scores)
        best_scores = np.copy(best_scores)

        # keep only best n particles
        best_particles = self._determineBestNParticles(
            scores=scores,
            best_scores=best_scores,
        )

        best_n_arr_list = []
        for arr in arr_list:
            best_n_arr_list.append(np.copy(arr[best_particles]))

        return best_n_arr_list, best_particles
    
    def _determineBestNParticles(
        self,
        scores:np.ndarray,
        best_scores:np.ndarray,
    ):
        """
        Determine best N particles.
        Args:
            scores: particle scores; numpy array of shape (N, T)
            best_scores: best scores; numpy array of shape (N,)
        Returns:
            best_particles: best particles; numpy array of shape (N,)
        """
        # return the best N particles from the last iteration if algorithm did not yet converge
        if self.converged_since_n_iters <= 0:
            best_particles = np.argsort(best_scores)
            best_particles = best_particles[:self.keep_best_n_particles]
            return best_particles
        
        # return mean of the best N particles from the last iteration if algorithm converged
        scores_mean = np.nanmean(scores[:,-self.converged_since_n_iters:], axis=1)
        best_particles = np.argsort(scores_mean)
        best_particles = best_particles[:self.keep_best_n_particles]
        return best_particles

    def _readPosData(
            self,
            read_vel=False,
        ):
        """
        Read position data from data_dir.
        Args:
            read_vel: read velocity instead of position; bool
        Returns:
            pos: particle positions; numpy array of shape (N, T, M)
            scores: particle scores; numpy array of shape (N, T)
            parameters: dictionary of parameters { column_index: parameter_name}; dictionary { int: str }
        """
        # read parameters and number of iterations
        if read_vel:
            files = [f'pso_vel_{i}.csv' for i in range(self.num_particles)]
        else:
            files = [f'pso_pos_{i}.csv' for i in range(self.num_particles)]
        df = pd.read_csv(os.path.join(self.data_dir, files[0]))
        columns = df.columns.to_list()
        if not read_vel:
            columns.remove('score')
            columns.remove('time')
            columns.remove('iteration')
        parameters = { i:param for i, param in enumerate(columns) }

        # read data
        pos = np.full((self.num_particles, len(df), len(columns)), np.nan)
        if not read_vel:
            scores = np.full((self.num_particles, len(df)), np.nan)

        for i, file in enumerate(files):
            df = pd.read_csv(os.path.join(self.data_dir, file))
            pos_temp = df[columns].to_numpy()
            pos[i, :len(pos_temp), :] = pos_temp

            if not read_vel:
                scores_temp = df[['score']].to_numpy()
                scores[i, :len(scores_temp)] = scores_temp.flatten()

        # verify that each column has at maximum one nan
        nans = np.sum(np.isnan(pos), axis=1)
        assert np.all(nans <= 1), f"Each column should have at maximum one nan, but found {nans}."

        if not read_vel:
            return pos, scores, parameters
        else:
            return pos, parameters
    
    def _readBestPosData(
            self,
        ):
        """
        Read position data from data_dir.
        Returns:
            pos: particle positions; numpy array of shape (N, M)
            scores: particle scores; numpy array of shape (N,)
            parameters: dictionary of parameters { column_index: parameter_name}; dictionary { int: str }
        """
        # read parameters and number of iterations
        files = [f'pso_best_pos_{i}.csv' for i in range(self.num_particles)]
        df = pd.read_csv(os.path.join(self.data_dir, files[0]))
        columns = df.columns.to_list()
        columns.remove('best_score')
        columns.remove('best_count')
        parameters = { i:param for i, param in enumerate(columns) }

        # read data
        best_pos = np.zeros((len(files), len(columns)))
        best_scores = np.zeros((len(files)))
        best_iters = np.zeros((len(files)), dtype=int)
        for i, file in enumerate(files):
            df = pd.read_csv(os.path.join(self.data_dir, file))
            pos_temp = df[columns].to_numpy()
            scores_temp = df[['best_score']].to_numpy().flatten()

            best_pos[i] = pos_temp[-1, :]
            best_scores[i] = scores_temp[-1]
            best_iters[i] = np.argmax(scores_temp == best_scores[i])

        return best_pos, best_scores, best_iters, parameters
    
    def _readHparamsLims(
        self,
    ):
        """
        Read hparams lims from data_dir.
        Returns:
            hparams_lims: dictionary of hparams limits { hparam_name: [min, max] }; dictionary { str: [float, float] }
        """
        # read json file
        hparams_lims_file = os.path.join(self.data_dir, 'hparams_lims.json')

        hparams_lims_temp = {}
        with open(hparams_lims_file) as f:
            hparams_lims_temp = json.load(f)

        # flatten dictionary
        hparams_lims = {}
        for elements in hparams_lims_temp.values():
            for hparam, lim in elements.items():
                hparams_lims[hparam] = lim

        return hparams_lims
    




