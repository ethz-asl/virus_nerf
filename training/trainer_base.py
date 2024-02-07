import os
import torch
import numpy as np
import pandas as pd
import taichi as ti
from alive_progress import alive_bar
from contextlib import nullcontext

from args.args import Args
from modules.networks import NGP
from modules.rendering import render
from helpers.geometric_fcts import createScanPos
from datasets.dataset_base import DatasetBase
from datasets.dataset_rh import DatasetRH
from datasets.dataset_ethz import DatasetETHZ


class TrainerBase():
    def __init__(
        self,
        hparams_file=None,
        args:Args=None,
        train_dataset:DatasetBase=None,
        test_dataset:DatasetBase=None,
    ) -> None:

        # get hyper-parameters and other variables
        if args is None:
            self.args = Args(
                file_name=hparams_file
            )
        else:
            self.args = args

        # initialize taichi
        taichi_init_args = {
            "arch": ti.cuda if torch.cuda.is_available() else ti.cpu,
            "random_seed": self.args.seed,
        }
        ti.init(**taichi_init_args)

        # datasets   
        if self.args.dataset.name == 'RH2':
            dataset = DatasetRH 
        elif self.args.dataset.name == 'ETHZ':   
            dataset = DatasetETHZ
        else:
            self.args.logger.error(f"Unknown dataset {self.args.dataset.name}")
        
        if train_dataset is None:
            self.train_dataset = dataset(
                args = self.args,
                split="train",
            ).to(self.args.device)
        else:
            self.train_dataset = train_dataset

        if test_dataset is None:
            self.test_dataset = dataset(
                args = self.args,
                split='test',
                scene=self.train_dataset.scene,
            ).to(self.args.device)
        else:
            self.test_dataset = test_dataset

        # model
        model_config = {
            'scale': self.args.model.scale,
            'pos_encoder_type': self.args.model.encoder_type,
            'levels': self.args.model.hash_levels,
            'max_res': self.args.model.hash_max_res, 
            'half_opt': False, # TODO: args
            'scene': self.train_dataset.scene,
            'args': self.args,
            'dataset': self.train_dataset,
        }
        self.model = NGP(**model_config).to(self.args.device)

        # load checkpoint if ckpt path is provided
        if self.args.model.ckpt_path:
            self._loadCheckpoint(ckpt_path=self.args.model.ckpt_path)

        # grid update interval
        if self.args.model.grid_type == 'ngp':
            self.grid_update_interval = self.args.ngp_grid.update_interval
        elif self.args.model.grid_type == 'occ':
            self.grid_update_interval = self.args.occ_grid.update_interval
        else:
            self.args.logger.error("Grid type not implemented!")

    def interfereDensityMap(
            self, 
            res_map:int, 
            height_w:float, 
            num_avg_heights:int,
            tolerance_w:float,
            threshold:float,
    ):
        """
        Evaluate slice density.
        Args:
            res_map: number of samples in each dimension (L); int
            height_w: height of slice in world coordinates (meters); float
            num_avg_heights: number of heights to average over (A); int
            tolerance_w: tolerance in world coordinates (meters); float
            threshold: threshold for density map; float
        Returns:
            density_map: density map of slice; array of shape (L, L)
        """
        # create position grid
        pos_avg = createScanPos(
            res_map=res_map,
            height_c=self.train_dataset.scene.w2c(pos=np.array([[0.0, 0.0, height_w]]), copy=True)[0,2],
            num_avg_heights=num_avg_heights,
            tolerance_c=self.train_dataset.scene.w2c(pos=tolerance_w, only_scale=True, copy=True),
            cube_min=self.test_dataset.scene.w2c_params["cube_min"],
            cube_max=self.test_dataset.scene.w2c_params["cube_max"],
            device=self.args.device,
        ) # (L*L*A, 3)

        # interfere density map
        density_map = torch.empty(0).to(self.args.device)
        for density_batch in self._batchifyDensity(
                pos=pos_avg,
                batch_size=self.args.eval.batch_size,
                test_time=True,
            ):
            density_map = torch.cat((density_map, density_batch), dim=0)

        density_map = density_map.detach().cpu().numpy().reshape(-1, num_avg_heights) # (L*L, A)
        density_map = np.nanmax(density_map, axis=1) # (L*L,)
        density_map = density_map.reshape(res_map, res_map) # (L, L)

        # threshold density map
        density_map_thr = np.zeros_like(density_map)
        density_map_thr[density_map < threshold] = 0.0
        density_map_thr[density_map >= threshold] = 1.0

        return density_map, density_map_thr # (L, L), (L, L)
    
    def _loadCheckpoint(
        self, 
        ckpt_path:str
    ):
        """
        Load checkpoint
        Args:
            ckpt_path: path to checkpoint; str
        """
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print("Load checkpoint from %s" % ckpt_path)

    def _saveModel(
        self,
    ):
        """
        Save model, args and logs
        """
        if not self.args.model.save:
            return
        
        print(f"Saving model to {self.args.save_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.args.save_dir, 'model.pth'),
        )
        self.args.saveJson()

        # remove empty logs
        del_keys = []
        for key in self.logs.keys():
            if len(self.logs[key]) == 0:
                del_keys.append(key)
        for key in del_keys:
            del self.logs[key]

        # save logs
        logs_df = pd.DataFrame(self.logs)
        logs_df.to_csv(os.path.join(self.args.save_dir, 'logs.csv'), index=False)
    
    def _batchifyRender(
        self,
        rays_o:torch.Tensor,
        rays_d:torch.Tensor,
        test_time:bool,
        batch_size:int,
    ):
        """
        Batchify rendering process.
        Args:
            rays_o: ray origins; tensor of shape (N, 3)
            rays_d: ray directions; tensor of shape (N, 3)
            test_time: test time rendering; bool
            batch_size: batch size; int 
        Yields:
            results: dict of rendered images of current batch
        """
        # calculate number of batches
        N = rays_o.shape[0]
        if N % batch_size == 0:
            num_batches = N // batch_size
        else:
            num_batches = N // batch_size + 1

        # render rays in batches
        with torch.no_grad() if test_time else nullcontext():
            with alive_bar(num_batches, bar = 'bubbles', receipt=False) as bar:
                for i in range(num_batches):
                    batch_start = i * batch_size
                    batch_end = min((i+1) * batch_size, N)
                    results = render(
                        self.model, 
                        rays_o=rays_o[batch_start:batch_end], 
                        rays_d=rays_d[batch_start:batch_end],
                        test_time=test_time,
                        exp_step_factor=self.args.exp_step_factor,
                    )
                    bar()
                    yield results

    def _batchifyDensity(
        self,
        pos:torch.Tensor,
        test_time:bool,
        batch_size:int,
    ):
        """
        Batchify density rendering process.
        Args:
            pos: ray origins; tensor of shape (N, 3)
            test_time: test time rendering; bool
            batch_size: batch size; int 
        Yields:
            sigmas: density of current batch; tensor of shape (N,)
        """
        # calculate number of batches
        N = pos.shape[0]
        if N % batch_size == 0:
            num_batches = N // batch_size
        else:
            num_batches = N // batch_size + 1

        # render rays in batches
        with torch.no_grad() if test_time else nullcontext():
            with alive_bar(num_batches, bar = 'bubbles', receipt=False) as bar:
                for i in range(num_batches):
                    batch_start = i * batch_size
                    batch_end = min((i+1) * batch_size, N)
                    sigmas = self.model.density(pos[batch_start:batch_end])
                    bar()
                    yield sigmas

    def _scanRays2scanMap(
        self,
        rays_o_w:np.array,
        depth:np.array,
        scan_angles:np.array,
        num_imgs:int,
    ):
        """
        Create scan maps for given rays and depths.
        Args:
            rays_o_w: ray origins in world coordinates (meters); numpy array of shape (N*M, 3)
            depth: depths in wolrd coordinates (meters); numpy array of shape (N*M,)
            scan_angles: scan angles; numpy array of shape (N*M,)
            num_imgs: number of images N; int
        Returns:
            scan_maps: scan maps; numpy array of shape (N, L, L)
        """
        L = self.args.eval.res_map
        N = num_imgs
        M = rays_o_w.shape[0] // N
        if rays_o_w.shape[0] % M != 0:
            self.args.logger.error(f"rays_o_w.shape[0]={rays_o_w.shape[0]} % M={M} != 0")
        
        # convert depth to position in world coordinate system and then to map indices
        pos = self.test_dataset.scene.depth2pos(rays_o=rays_o_w, scan_depth=depth, scan_angles=scan_angles) # (N*M, 2)
        idxs = self.test_dataset.scene.w2idx(pos=pos, res=L) # (N*M, 2)
        idxs = idxs.reshape(N, M, 2) # (N, M, 2)

        # create scan map
        scan_maps = np.zeros((N, L, L))
        for i in range(N):
            scan_maps[i, idxs[i,:,0], idxs[i,:,1]] = 1.0

        return scan_maps # (N, L, L)

    def _step2time(
        self,
        steps:np.array,
    ):
        """
        Convert steps to time by linear interpolating the logs.
        Args:
            steps: steps to convert; array of shape (N,)
        Returns:
            times: times of given steps; array of shape (N,)
        """
        if len(steps) == 0:
            return np.array([])
        
        slope = self.logs['time'][-1] / self.logs['step'][-1]
        return slope * steps
    
    def _time2step(
        self,
        times:np.array,
    ):
        """
        Convert time to steps by linear interpolating the logs.
        Args:
            times: times to convert; array of shape (N,)
        Returns:
            steps: steps of given times; array of shape (N,)
        """
        if len(times) == 0:
            return np.array([])
        
        slope = self.logs['step'][-1] / self.logs['time'][-1]
        return slope * times
        

