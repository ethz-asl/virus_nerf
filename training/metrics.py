import numpy as np
import torch
from abc import ABC, abstractmethod
from einops import rearrange

from torchmetrics import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)

from args.args import Args
from helpers.geometric_fcts import findNearestNeighbour

class Metrics():
    def __init__(
        self,
        args:Args,
        img_wh:tuple,
    ) -> None:
        """
        Metrics base class.
        Args:
            args: arguments; Args object
            img_wh: image width and height; tuple of int (2,)
        """
        self.args = args
        self.img_wh = img_wh

        self.val_psnr = PeakSignalNoiseRatio(
            data_range=1
        ).to(self.args.device)
        self.val_ssim = StructuralSimilarityIndexMeasure(
            data_range=1
        ).to(self.args.device)

    @abstractmethod
    def convertData(self):
        pass

    def evaluate(
        self, 
        data:dict,
        eval_metrics:list,
        convert_to_world_coords:bool=True, 
        copy:bool=True,
        num_test_pts:int=None,
    ) -> dict:
        """
        Evaluate metrics listed in eval_metrics.
        Args:
            data: data dictionary, containing one or multiple of the following keys:
                depth: predicted depth; either numpy array or torch tensor (N*M,)
                depth_gt: ground truth depth; either numpy array or torch tensor (N*M,)
                rays_o: origin of rays in world coordinates; numpy array (N*M, 3)
                scan_angles: scan angles in radians; numpy array (M,)
                rgb: predicted rgb; either numpy array or torch tensor (N*H*W, 3)
                rgb_gt: ground truth rgb; either numpy array or torch tensor (N*H*W, 3)
            eval_metrics: list of metrics to evaluate; list of str
            convert_to_world_coords: convert depth to world coordinates (meters); bool
            copy: whether or not to copy input arrays/tensors; bool
            num_test_pts: number of points to evaluate metrics on; int
        Returns:
            dict: dictionary containing the metrics; dict
        """
        # copy input arrays/tensors
        if copy:
            data = self._copyData(data=data)

        # check that all required data is provided
        self._checkData(data=data, eval_metrics=eval_metrics, num_test_pts=num_test_pts)

        # convert data to right format and coordinate system
        if 'depth' in data: # TODO: change condition and function
            data = self.convertData(
                data=data, 
                eval_metrics=eval_metrics, 
                convert_to_world_coords=convert_to_world_coords,
                num_test_pts=num_test_pts,
            )

        dict = {}
        for metric in eval_metrics:
            if metric == 'rmse':
                dict['rmse'] = self._rmse(depth=data['depth'], depth_gt=data['depth_gt'])
            elif metric == 'mae':
                dict['mae'] = self._mae(depth=data['depth'], depth_gt=data['depth_gt'])
            elif metric == 'mare':
                dict['mare'] = self._mare(depth=data['depth'], depth_gt=data['depth_gt'])
            elif metric == 'nn':
                nn_dists, mnn = self._nn(pos=data['pos'], pos_ref=data['pos_gt'], num_test_pts=num_test_pts, depth_gt=data['depth_gt'])
                dict['nn_dists'] = nn_dists
                dict['mnn'] = mnn
            elif metric == 'nn_inv':
                nn_dists_inv, mnn_inv = self._nn(pos=data['pos_gt'], pos_ref=data['pos'], num_test_pts=num_test_pts, depth_gt=data['depth_gt'])
                dict['nn_dists_inv'] = nn_dists_inv
                dict['mnn_inv'] = mnn_inv
            elif metric == 'rnn':
                rnn_dists, mrnn = self._rnn(pos=data['pos'], pos_ref=data['pos_gt'], num_test_pts=num_test_pts, depth_gt=data['depth_gt'])
                dict['rnn_dists'] = rnn_dists
                dict['mrnn'] = mrnn
            elif metric == 'rnn_inv':
                rnn_dists_inv, mrnn_inv = self._rnn(pos=data['pos_gt'], pos_ref=data['pos'], num_test_pts=num_test_pts, depth_gt=data['depth_gt'])
                dict['rnn_dists_inv'] = rnn_dists_inv
                dict['mrnn_inv'] = mrnn_inv
            elif metric == 'psnr':
                dict['psnr'] = self._psnr(rgb=data['rgb'], rgb_gt=data['rgb_gt'])
            elif metric == 'ssim':
                dict['ssim'] = self._ssim(rgb=data['rgb'], rgb_gt=data['rgb_gt'])
            else:
                self.args.logger.warning(f"metric {metric} not implemented")

        return dict
    
    def nn(
        self, 
        pos:np.array, 
        pos_ref:np.array,
        depths:np.array,
        depths_gt:np.array,
        num_points:int,
        ref_pos_is_gt:bool,
    ):
        """
        Calculate nearest neighbour distance between pos_w and pos_w_gt
        Args:
            pos: position in world coordinate system; either numpy array (N*K, 2)
            pos_ref: reference position in world coordinate system; either numpy array (N*M, 2)
            depths: predicted depth; either numpy array
                        - if ref_pos_is_gt=True: shape = (N*K,)
                        - if ref_pos_is_gt=False: shape = (N*M,)
            depths_gt: ground truth depth; either numpy array
                        - if ref_pos_is_gt=True: shape = (N*M,)
                        - if ref_pos_is_gt=False: shape = (N*K,)
            num_points: number of points (N); int
            ref_pos_is_gt: whether or not the reference position the ground truth is; bool
        Returns:
            nn_dists: nearest neighbour distances; either numpy array (N*K,)
            nn_mean: mean of nearest neighbour distances {zone:mnn}; dict {str: float}
            nn_median: median of nearest neighbour distances {zone:mrnn}; dict {str: float}
            nn_inlier: ratio of points with a NN-distance smaller than a threshold {zone:inlier_ratio}; dict {str: float}
            nn_outlier_too_close: ratio of points with a NN-distance larger than a threshold but the measurement is closer 
                                    than the ground truth {zone:outlier_ratio}; dict {str: float}
        """
        pos = np.copy(pos)
        pos_ref = np.copy(pos_ref)
        depths = np.copy(depths)
        depths_gt = np.copy(depths_gt)
        
        N = num_points
        K = pos.shape[0] // N
        M = pos_ref.shape[0] // N

        pos = pos.reshape(N, K, 2) # (N, K, 2)
        pos_ref = pos_ref.reshape(N, M, 2) # (N, M, 2)
        if ref_pos_is_gt:
            depths = depths.reshape(N, K) # (N, K)
            depths_gt = depths_gt.reshape(N, M) # (N, M)
        else:
            depths = depths.reshape(N, M) # (N, M)
            depths_gt = depths_gt.reshape(N, K) # (N, K)
        
        
        if ref_pos_is_gt:
            nn_depths = depths # (N, K)
            nn_depths_gt = np.zeros((N, K), dtype=np.int32) # (N, K)
        else:
            nn_depths = np.zeros((N, K), dtype=np.int32) # (N, K)
            nn_depths_gt = depths_gt # (N, K)
        
        nn_dists = np.zeros((N, K)) # (N, K)
        for i in range(N):
            idxs, dists = findNearestNeighbour(
                array1=pos[i], 
                array2=pos_ref[i],
                ignore_nan=True,
            ) # (K,), (K,)
            nn_dists[i] = dists

            if ref_pos_is_gt:
                nn_depths_gt[i] = np.where((idxs < 0), np.nan, depths_gt[i, idxs]) # (K,)
            else:
                nn_depths[i] = np.where((idxs < 0), np.nan, depths[i, idxs]) # (K,)

        
        nn_dists = nn_dists.flatten() # (N*K,)
        nn_depths = nn_depths.flatten() # (N*K,)
        nn_depths_gt = nn_depths_gt.flatten() # (N*K,)

        nn_mean = {}
        nn_median = {}
        nn_inlier = {}
        nn_outlier_too_close = {}
        for zone, min_max in self.args.eval.zones.items():
            # mask containing valid points (not nan) that are in current zone
            valid_mask = (~np.isnan(nn_dists)) & (nn_depths_gt >= min_max[0]) & (nn_depths_gt <= min_max[1]) # (N*K,)

            # mask containing inlier and outlier points
            inlier_mask = (nn_dists < self.args.eval.inlier_threshold) # (N*K,)

            # mask containing points where the measurement is closer than nearest neighbour of ground truth
            too_close_mask = (nn_depths < nn_depths_gt) # (N*K,)

            nn_mean[zone] = np.nanmean(nn_dists[valid_mask])
            nn_median[zone] = np.nanmedian(nn_dists[valid_mask])
            nn_inlier[zone] = np.sum(valid_mask & inlier_mask) / np.sum(valid_mask)
            nn_outlier_too_close[zone] = np.sum(valid_mask & ~inlier_mask & too_close_mask) / np.sum(valid_mask)

            if self.args.training.debug_mode:
                nn_outlier_tot_far = np.sum(valid_mask & ~inlier_mask & ~too_close_mask) / np.sum(valid_mask)
                if (nn_inlier[zone] + nn_outlier_too_close[zone] + nn_outlier_tot_far - 1.0) > 1e-6:
                    self.args.logger.error(f"nn_inlier + nn_outlier_too_close + nn_outlier_tot_far = " 
                                           f"{nn_inlier[zone] + nn_outlier_too_close[zone] + nn_outlier_tot_far} != 1.0")

        return nn_dists, nn_mean, nn_median, nn_inlier, nn_outlier_too_close
    
    def nn_dists(
        self, 
        pos:np.array, 
        pos_ref:np.array,
        num_points:int,
    ):
        """
        Calculate nearest neighbour distance between pos_w and pos_w_gt
        Args:
            pos: position in world coordinate system; either numpy array (N*K, 2)
            pos_ref: reference position in world coordinate system; either numpy array (N*M, 2)
            depths_gt: ground truth depth; either numpy array (N*M,)
            num_points: number of points (N); int
        Returns:
            nn_idxs: nearest neighbour indices; either numpy array (N*K,)
            nn_dists: nearest neighbour distances; either numpy array  (N*K,)
        """
        N = num_points
        K = pos.shape[0] // N
        M = pos_ref.shape[0] // N

        pos = pos.reshape(N, K, 2) # (N, K, 2)
        pos_ref = pos_ref.reshape(N, M, 2)
        
        nn_idxs = np.zeros((N, K), dtype=np.int32) # (N, K)
        nn_dists = np.zeros((N, K)) # (N, K)
        for i in range(N):
            idxs, dists = findNearestNeighbour(
                array1=pos[i], 
                array2=pos_ref[i],
                ignore_nan=True,
            )
            nn_dists[i] = dists
            nn_idxs[i] = idxs

        nn_idxs = nn_idxs.flatten() # (N*K,)
        nn_dists = nn_dists.flatten() # (N*K,)
        return nn_idxs, nn_dists
    
    def _rmse(
        self,
        depth, 
        depth_gt
    ):
        """
        Calculate Root Mean Squared Error (RMSE) between depth and depth_gt
        Args:
            depth: predicted depth; either numpy array (N*M,) or torch tensor (N*M,)
            depth_gt: ground truth depth; either numpy array (N*M,) or torch tensor (N*M,)
        Returns:
            rmse: root mean squared error for zones; dict {str: float}
        """
        rmse = {}
        for zone, min_max in self.args.eval.zones.items():
            mask = (depth_gt >= min_max[0]) & (depth_gt <= min_max[1])
            if torch.is_tensor(depth):
                rmse[zone] = torch.nanmean((depth[mask] - depth_gt[mask])**2).item()
            else:
                rmse[zone] = np.nanmean((depth[mask] - depth_gt[mask])**2)
        return rmse

    def _mae(
        self, 
        depth, 
        depth_gt
    ):
        """
        Calculate Mean Absolute Error (MAE) between depth and depth_gt
        Args:
            depth: predicted depth; either numpy array or torch tensor (N*M,)
            depth_gt: ground truth depth; either numpy array or torch tensor (N*M,)
        Returns:
            mae: mean absolute error for zones; dict {str: float}
        """
        mae = {}
        for zone, min_max in self.args.eval.zones.items():
            mask = (depth_gt >= min_max[0]) & (depth_gt <= min_max[1])
            if torch.is_tensor(depth):
                mae[zone] = torch.nanmean(torch.abs(depth[mask] - depth_gt[mask])).item()
            else:
                mae[zone] = np.nanmean(np.abs(depth[mask] - depth_gt[mask]))
        return mae
    
    def _mare(
        self, 
        depth, 
        depth_gt
    ):
        """
        Calculate Mean Absolute Relative Error (MARE) between depth and depth_gt
        Args:
            depth: predicted depth; either numpy array or torch tensor (N*M,)
            depth_gt: ground truth depth; either numpy array or torch tensor (N*M,)
        Returns:
            mare: mean absolute relative error for zones; dict {str: float}
        """
        mare = {}
        for zone, min_max in self.args.eval.zones.items():
            mask = (depth_gt >= min_max[0]) & (depth_gt <= min_max[1])
            if torch.is_tensor(depth):
                mare[zone] = torch.nanmean(torch.abs((depth[mask] - depth_gt[mask])/ depth_gt[mask])).item()
            else:
                mare[zone] = np.nanmean(np.abs((depth[mask] - depth_gt[mask])/ depth_gt[mask]))
        return mare
    

    
    def _rnn(
        self, 
        pos:np.array, 
        pos_ref:np.array,
        num_test_pts:int,
        depth_gt:np.array,
    ):
        """
        Calculate nearest neighbour distance between pos_w and pos_w_gt
        Args:
            pos: position in world coordinate system; either numpy array or torch tensor (N, M, 2)
            pos_gt: reference position in world coordinate system; either numpy array or torch tensor (N, M, 2)
            num_test_pts: number of test points (N); int
            depth_gt: ground truth depth; either numpy array or torch tensor (N*M)
        Returns:
            nn_dists: nearest neighbour distances; either numpy array or torch tensor (N,M)
            mnn: mean of nearest neighbour distances; float
        """
        rnn_dists = np.zeros((num_test_pts, pos.shape[1]))
        for i in range(num_test_pts):
            _, dists = findNearestNeighbour(
                array1=pos[i], 
                array2=pos_ref[i],
                ignore_nan=True,
            )
            rnn_dists[i] = dists

        rnn_dists = rnn_dists.flatten()

        mrnn = {}
        for zone, min_max in self.args.eval.zones.items():
            mask = (depth_gt >= min_max[0]) & (depth_gt <= min_max[1]) 
            mrnn[zone] = np.nanmean(rnn_dists[mask] / depth_gt[mask])

        return rnn_dists, mrnn

    def _psnr(
            self,
            rgb:torch.tensor,
            rgb_gt:torch.tensor,
    ):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between rgb and rgb_gt
        Args:
            rgb: predicted rgb; either numpy array or torch tensor (N*H*W, 3)
            rgb_gt: ground truth rgb; either numpy array or torch tensor (N*H*W, 3)
        Returns:
            psnr: average peak signal-to-noise ratio; float
        """
        W, H = self.img_wh
        if rgb.shape[0] % (W*H) != 0:
            self.args.logger.error(f"rgb.shape[0] = {rgb.shape[0]} must be divisible by W*H = {W*H}")
        num_imgs = rgb.shape[0] // (W*H)

        if num_imgs == 0:
            return 0.0

        test_psnrs = []
        for i in range(num_imgs):
            rgb_img = rearrange(rgb[i*H*W:(i+1)*W*H], '(h w) c -> 1 c h w', h=H) # TODO: optimize
            rgb_img_gt = rearrange(rgb_gt[i*H*W:(i+1)*W*H], '(h w) c -> 1 c h w', h=H)

            self.val_psnr(rgb_img, rgb_img_gt)
            test_psnrs.append(self.val_psnr.compute())
            self.val_psnr.reset()
        psnrs = sum(test_psnrs) / len(test_psnrs)
        return psnrs.detach().cpu().item()

    def _ssim(
            self,
            rgb:torch.tensor,
            rgb_gt:torch.tensor,
    ):
        """
        Calculate Structural Similarity Index Measure (SSIM) between rgb and rgb_gt.
        Args:
            rgb: predicted rgb; either numpy array or torch tensor (N*H*W, 3)
            rgb_gt: ground truth rgb; either numpy array or torch tensor (N*H*W, 3)
        Returns:
            ssim: average structural similarity index measure; float
        """
        W, H = self.img_wh
        if rgb.shape[0] % (W*H) != 0:
            self.args.logger.error(f"rgb.shape[0] = {rgb.shape[0]} must be divisible by W*H = {W*H}")
        num_imgs = rgb.shape[0] // (W*H)

        if num_imgs == 0:
            return 0.0

        test_ssims = []
        for i in range(num_imgs):
            rgb_img = rearrange(rgb[i*H*W:(i+1)*W*H], '(h w) c -> 1 c h w', h=H) # TODO: optimize
            rgb_img_gt = rearrange(rgb_gt[i*H*W:(i+1)*W*H], '(h w) c -> 1 c h w', h=H)

            self.val_ssim(rgb_img, rgb_img_gt)
            test_ssims.append(self.val_ssim.compute())
            self.val_ssim.reset()

        ssims = sum(test_ssims) / len(test_ssims)
        return ssims.detach().cpu().item()
    
    def _copyData(
            self,
            data:dict,
    ):
        """
        Copy data dictionary.
        Args:
            data: data dictionary
        Returns:
            data_copy: data dictionary
        """
        data_copy = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                data_copy[key] = value.clone().detach()
            else:
                data_copy[key] = np.copy(value)
        return data_copy
    
    def _checkData(
            self,
            data:dict,
            eval_metrics:list,
            num_test_pts:int,
    ):
        """
        Check if data dictionary contains all required keys.
        Args:
            data: data dictionary
            eval_metrics: list of metrics to evaluate; list of str
        """
        if 'nn' in eval_metrics:
            if (not 'rays_o' in data) or (not 'scan_angles' in data) or (not 'depth' in data) \
                or (not 'depth_gt' in data) or (num_test_pts is None):
                print("WARNING: rays_o, scan_angles, depth, depth_gt and num_test_pts must be provided for metric 'nn'")
                eval_metrics.remove('nn')
        
        if ('mse' in eval_metrics):
            if (not 'depth' in data) or (not 'depth_gt' in data):
                print("WARNING: depth and depth_gt must be provided for metric 'mse'")
                eval_metrics.remove('mse')

        if ('mae' in eval_metrics):
            if (not 'depth' in data) or (not 'depth_gt' in data):
                print("WARNING: depth and depth_gt must be provided for metric 'mae'")
                eval_metrics.remove('mae')

        if ('mare' in eval_metrics):
            if (not 'depth' in data) or (not 'depth_gt' in data):
                print("WARNING: depth and depth_gt must be provided for metrics 'mare'")
                eval_metrics.remove('mare')

        if ('ssim' in eval_metrics):
            if (not 'rgb' in data) or (not 'rgb_gt' in data):
                print("WARNING: rgb and rgb_gt must be provided for metric 'ssim'")
                eval_metrics.remove('ssim')
        
        if ('psnr' in eval_metrics):
            if (not 'rgb' in data) or (not 'rgb_gt' in data):
                print("WARNING: rgb and rgb_gt must be provided for metric 'psnr'")
                eval_metrics.remove('psnr')

    

# def nnNumpy(
#             self, 
#             array1:np.array, 
#             array2:np.array
#         ):
#         """
#         Find the closest points in array2 for each point in array1
#         and return the indices of array2 for each point in array1.
#         Args:
#             array1: array of float (N, 2/3)
#             array2: array of float (M, 2/3)
#         Returns:
#             nn_idxs: indices of nearest neighbours from array2 with respect to array1; array of int (N,)
#             nn_dists: distances of nearest neighbours from array2 with respect to array1; array of float (N,)
#         """
#         # downsample arrays
#         array1 = np.copy(array1.astype(np.float32))
#         array2 = np.copy(array2.astype(np.float32))

#         # determine nearest neighbour indices and distances
#         dists = np.linalg.norm(array2[:, np.newaxis] - array1, axis=2) # (M, N)
#         nn_idxs = np.argmin(dists, axis=0) # (N,)
#         nn_dists = np.min(dists, axis=0) # (N,)
        
#         return nn_idxs, nn_dists
    
#     def nnTorch(
#             self, 
#             tensor1:torch.tensor, 
#             tensor2:torch.tensor,
#         ):
#         """
#         Find the closest points in array2 for each point in array1
#         and return the indices of array2 for each point in array1.
#         Args:
#             array1: tensor of float (N, 2/3)
#             array2: tensor of float (M, 2/3)
#         Returns:
#             nn_idxs: indices of nearest neighbours from tensor2 with respect to tensor1; array of int (N,)
#             nn_dists: distances of nearest neighbours from tensor2 with respect to tensor1; array of float (N,)
#         """
#         # downsample arrays
#         tensor1 = np.copy(tensor1.astype(np.float32))
#         tensor2 = np.copy(tensor2.astype(np.float32))

#         # determine nearest neighbour indices and distances
#         dists = torch.linalg.norm(tensor2[:, np.newaxis] - tensor1, dim=2) # (M, N)
#         nn_idxs = torch.argmin(dists, dim=0) # (N,)
#         nn_dists = torch.min(dists, dim=0) # (N,)
        
#         return nn_idxs, nn_dists