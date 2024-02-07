import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from optimization.particle_swarm_optimization_wrapper import ParticleSwarmOptimizationWrapper
from optimization.metric import Metric



class Cmaps():
    def __init__(
        self,
        num_cmaps:int,
        norm_min:float,
        norm_max:float,
        skip_bright_colors:bool=False,
    ) -> None:
        cmap_names = ["Blues", "Greens", "Purples", "Oranges", "Reds"]
        self.cmaps = [matplotlib.colormaps[cmap_names[n%len(cmap_names)]] for n in range(num_cmaps)]

        self.skip_bright_colors = skip_bright_colors
        if self.skip_bright_colors:
            self.norm_delta = (norm_max - norm_min) / 2
        self.norm = matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max + self.norm_delta)

    def __call__(
        self,
        cmap_idx:int,
        val:float,
    ):
        """
        Determine color based on cmap and value.
        Args:
            cmap_idx: index of colormap; int
            val: value to determine color; float
        Returns:
            color: color; np.array (4,)
        """
        if self.skip_bright_colors:
            val += self.norm_delta
        return self.cmaps[cmap_idx](self.norm(val))
    

class Plotter():
    def __init__(
        self,
        num_axes:int,
    ) -> None:
        # determine number of rows and columns
        squares = np.arange(1, num_axes+1)**2
        for s in squares:
            if s >= num_axes:
                self.num_rows = int(np.sqrt(s))
                self.num_cols = int(np.sqrt(s))
                break

        # create figure
        self.fig, self.axes = plt.subplots(
            ncols=self.num_cols, 
            nrows=self.num_rows, 
            figsize=(max(9, 3+3*self.num_cols), max(7, 1+3*self.num_rows))
        )

        # delete unused axes
        if num_axes > 1:
            self.axes = self.axes.flatten()
            for ax in self.axes[num_axes:]:
                ax.remove()

    def show(
        self,
        save_path:str=None,
    ):
        """
        Show and save figure.
        Args:
            save_path: path to save figure; str
        """
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.1, 0.05, 0.8]) # [left, bottom, width, height]
        self.fig.colorbar(self.im, cax=cbar_ax)
        plt.show()

        if save_path is not None:
            self.fig.savefig(save_path)
    
    def plot2D(
        self,
        pso:ParticleSwarmOptimizationWrapper,
        metric:Metric,
        ax_idx:int,
        res:int=64,
    ):
        """
        Plot 2D optimization.
        Args:
            pso: particle swarm optimization; ParticleSwarmOptimizationWrapper
            metric: metric; Metric
            ax_idx: index of axis; int
            res: resolution of plot; int
        """
        if self.num_cols*self.num_rows > 1:
            ax = self.axes[ax_idx]
        else:
            ax = self.axes

        pos, vel, best_pos, best_score = self._loadData(
            pso=pso,
        ) # (N, M, L), (N, M, L), (N, M, L)
        N = pos.shape[0] # number of particles
        M = pos.shape[1] # number of hparams
        L = pos.shape[2] # number of iterations

        # interfere gaussian
        m1, m2 = np.meshgrid(
            np.linspace(pso.hparams_lims[0, 0], pso.hparams_lims[0, 1], num=res),
            np.linspace(pso.hparams_lims[1, 0], pso.hparams_lims[1, 1], num=res),
            indexing='ij',
        )
        X = np.stack((m1.flatten(), m2.flatten()), axis=-1)
        scores = metric(
            X=X,
        )
        scores = scores.reshape((res, res))

        extent = [pso.hparams_lims[0, 0], pso.hparams_lims[0, 1], pso.hparams_lims[1,0], pso.hparams_lims[1, 1]]
        self.im = ax.imshow(scores.T, origin='lower', extent=extent, cmap='Greys', vmin=0, vmax=1)

        cmaps = Cmaps(
            num_cmaps=N,
            norm_min=0,
            norm_max=L-1,
            skip_bright_colors=True,
        )
        for n in range(N):
            for l in range(L-1):
                ax.plot([pos[n, 0, l], pos[n, 0, l+1]], [pos[n, 1, l], pos[n, 1, l+1]], 
                        color=cmaps(n, l), linewidth=2)

        ax.scatter(metric.centre[0], metric.centre[1], color="black", s=200, marker='*') 
        for n in range(N):
            ax.scatter(best_pos[n, 0, -1], best_pos[n, 1, -1], color=cmaps(n, L-2), s=100, marker='*') 
            ax.scatter(pos[n, 0, 0], pos[n, 1, 0], color=cmaps(n, 0), s=10) 
            arrow = 0.02 * vel[n, :, -1] / np.linalg.norm(vel[n, :, -1])
            ax.arrow(pos[n, 0, -1], pos[n, 1, -1], arrow[0], arrow[1], color=cmaps(n, L-2), linewidth=2, 
                     head_width=0.02, head_length=0.02)

        hparams_order_inv = {}
        for hparam in pso.hparams_order.keys():
            if pso.hparams_order[hparam] in hparams_order_inv.keys():
                print("ERROR: test_psoGauss: more than one parameter with order 0")
            hparams_order_inv[pso.hparams_order[hparam]] = hparam
        if ax_idx >= (self.num_rows-1)*self.num_cols:
            ax.set_xlabel(str(hparams_order_inv[0]))
        else:
            ax.set_xticks([])   
        if ax_idx % self.num_cols == 0:
            ax.set_ylabel(str(hparams_order_inv[1]))
        else:
            ax.set_yticks([])

        best_idx = np.argmin(best_score[:,-1])
        ax.set_title(f"score={best_score[best_idx,-1]:.3f}, "
                    + f"dist={np.linalg.norm(metric.centre - best_pos[best_idx,:,-1]):.2f}")
        
        if self.num_cols*self.num_rows > 1:
            self.axes[ax_idx] = ax
        else:
            self.axes = ax

    def _loadData(
        self,
        pso:ParticleSwarmOptimizationWrapper,
    ):
        """
        Load data from files.
        Args:
            pso: particle swarm optimization; ParticleSwarmOptimizationWrapper
        Returns:
            pos: position of particles; np.array (N, M, L)
            vel: velocity of particles; np.array (N, M, L)
            best_pos: best position of particles; np.array (N, M, L)
            best_score: best score of particles; np.array (N, L)
        """
        N = pso.N # number of particles
        M = pso.M # number of hyperparameters
        pos_dict = pso._loadStateFromFile(
            file_path=pso.pos_files[-1],
            return_last_row=False,
        ) # dict of lists of floats
        T = pos_dict["iteration"][-1] # number of iterations
        L = int(np.ceil(T / N) + 1) # number of iterations per particle, +1 for initial position

        pos = np.zeros((N, M, L))
        vel = np.zeros((N, M, L))
        best_pos = np.zeros((N, M, L))
        best_score = np.zeros((N, L))
        for i in range(N):
            # load data
            pos_dict = pso._loadStateFromFile(
                file_path=pso.pos_files[i],
                return_last_row=False,
            ) # dict of lists of floats
            del pos_dict["score"]
            del pos_dict["time"]
            del pos_dict["iteration"]     

            best_pos_dict = pso._loadStateFromFile(
                file_path=pso.best_pos_files[i],
                return_last_row=False,
            ) # dict of lists of floats
            best_score[i] = np.array(best_pos_dict["best_score"])
            del best_pos_dict["best_score"]
            del best_pos_dict["best_count"]

            vel_dict = pso._loadStateFromFile(
                file_path=pso.vel_files[i],
                return_last_row=False,
            ) # dict of lists of floats

            # convert to np.array
            pos[i] = pso._nameDict2hparam(
                name_dict=pos_dict,
            ) # np.array (L, M)

            vel[i] = pso._nameDict2hparam(
                name_dict=vel_dict,
            ) # np.array (L, M)

            best_pos[i] = pso._nameDict2hparam(
                name_dict=best_pos_dict,
            ) # np.array (L, M)

        return pos, vel, best_pos, best_score
