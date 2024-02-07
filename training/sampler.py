import numpy as np
import torch
import copy
import sys

from args.args import Args


class Sampler():
    def __init__(
            self,
            args:Args,
            dataset_len:int,
            img_wh:tuple,
            sensors_dict:dict=None,
            times:torch.Tensor=None,
    ) -> None:
        """
        Args:
            args: hyper parameters; instance of Args class
            dataset_len: number of images in dataset; int
            img_wh: image width and height; tuple of int (2,)
            sensors_dict: dictionary of sensor models; dict
            times: time stamps of images; tensor of float (N,)
        """
        self.args = args
        self.dataset_len = dataset_len
        self.img_wh = img_wh
        self.sensors_dict = sensors_dict
        self.times = times

        self.rgn = np.random.default_rng(seed=self.args.seed)

    def __call__(
        self,
        batch_size:int,
        sampling_strategy:dict,
        elapse_time:float,
    ):
        """
        Sample images and pixels/rays.
        Args:
            batch_size: number of samples; int
            sampling_strategy: sampling strategy used; dict
                'imgs': str
                    'all': sample from all images
                    'same': sample from the one image
                'rays': strategy used to sample rays or dictionary of strategies {strategy: share}; str or dict {str: float}
                    'random': sample random pixels
                    'entire_img': sample all pixels
                    'closest': sample closes pixels predicted by USS
                    'valid_uss': sample random pixels with valid depth of USS
                    'valid_tof': sample random pixels with valid depth of ToF
            elapse_time: elapse time since training start in seconds; float
        Returns:
            img_idxs: indices of images to be used for training; tensor of int64 (batch_size,)
            pix_idxs: indices of pixels to be used for training; tensor of int64 (batch_size,)
        """
        img_idxs = self._imgIdxs(
            batch_size=batch_size,
            img_strategy=sampling_strategy["imgs"],
            elapse_time=elapse_time,
        )
        pix_idxs = self._pixIdxs(
            pix_strategy=sampling_strategy["pixs"],
            img_idxs=img_idxs,
        )
        return img_idxs, pix_idxs
    
    def getValidImgIdxs(
        self,
        elapse_time:float,
    ):
        """
        Simulate real-time behaviour by sampling only from measurements that are already made.
        Args:
            elapse_time: elapse time since training start in seconds; float
        Return:
            valid_img_idxs: valid image indices; tensor of ints (N,)
        """
        valid_img_idxs = torch.arange(self.dataset_len, device=self.args.device, dtype=torch.int32)

        if self.args.training.real_time_simulation:
            mask = (self.times <= elapse_time)
            valid_img_idxs = valid_img_idxs[mask]

        # if self.args.training.debug_mode:
        if valid_img_idxs.shape[0] == 0:
            self.args.logger.error(f"no valid images found")
            sys.exit()
            
        return valid_img_idxs
    
    def _imgIdxs(
            self,
            batch_size:int,
            img_strategy:str,
            elapse_time:float,
    ):
        """
        Sample image indices.
        Args:
            batch_size: number of samples; int
            img_strategy: image sampling strategy; str
            elapse_time: elapse time since training start in seconds; float
        Returns:
            img_idxs: indices of images to be used for training; tensor of int64 (batch_size,)
        """
        valid_img_idxs = self.getValidImgIdxs(
            elapse_time=elapse_time,
        )

        if img_strategy == "all":
            idxs = torch.randint(0, valid_img_idxs.shape[0], size=(batch_size,), device=self.args.device, dtype=torch.int32)
            return valid_img_idxs[idxs]
        
        if img_strategy == "same":
            idx = torch.randint(0, valid_img_idxs.shape[0], size=(1,), device=self.args.device, dtype=torch.int32)
            img_idx = valid_img_idxs[idx]
            return img_idx * torch.ones(batch_size, device=self.args.device, dtype=torch.int32)               
        
        self.args.logger.error(f"image sampling strategy must be either 'all' or 'same'"
                + f" but is {self.args.training.sampling_strategy['imgs']}")

    def _pixIdxs(
            self,
            pix_strategy:dict,
            img_idxs:torch.Tensor=None,
    ):
        """
        Sample pixel indices.
        Args:
            pix_strategy: ray sampling strategy or multiple strategies {strategy: share}; str or dict {str: float}
            img_idxs: indices of images needed if ray sampling strategy is 'closest'; tensor of int64 (batch_size,)
        Returns:
            pix_idxs: indices of pixels to be used for training; tensor of int64 (batch_size,)
        """
        B = img_idxs.shape[0]
        pix_strategy = copy.deepcopy(pix_strategy)

        # return all pixels if strategy is 'entire_img'
        if pix_strategy == "entire_img":
            return self._pixStrategyEntireImg()

        # convert string to dictionary with one entry
        if isinstance(pix_strategy, str):
            pix_strategy = {pix_strategy: 1.0}

        # verify that sum of shares is between 0 and 1
        if self.args.training.debug_mode:
            share_sum = sum(pix_strategy.values())
            if share_sum < 0.0 or share_sum > 1.0:
                self.args.logger.error(f"ray sampling strategy shares must be between 0 and 1 "
                                       f"but sum is {sum(pix_strategy.values())}")
                return None

        # convert strategy share to batch size 
        B_sum = 0
        for strategy, share in pix_strategy.items():
            B = int(share*img_idxs.shape[0])
            pix_strategy[strategy] = B
            B_sum += B

        # assign all remaining pixels to 'random'
        B_rest = int(img_idxs.shape[0] - B_sum)
        if B_rest > 0:
            pix_strategy["random"] = B_rest

        # sample pixels for each strategy
        B_sum = 0
        pix_idxs = -1 * torch.ones(img_idxs.shape[0], device=self.args.device, dtype=torch.int32)
        for strategy, B in pix_strategy.items():
            if strategy == "random":
                pix_idxs_temp = self._pixStrategyRandom(
                    B=B,
                )
            elif strategy == "closest":
                pix_idxs_temp = self._pixStrategyClosest(
                    img_idxs=img_idxs[B_sum:B_sum+B],
                )
            elif strategy == "valid_uss":
                pix_idxs_temp = self._pixStrategyValidDepth(
                    B=B,
                    sensor_type="USS",
                )
            elif strategy == "valid_tof":
                pix_idxs_temp = self._pixStrategyValidDepth(
                    B=B,
                    sensor_type="ToF",
                )
            else:
                self.args.logger.error(f"ray sampling strategy = {strategy} not implemented")

            pix_idxs[B_sum:B_sum+B] = pix_idxs_temp
            B_sum += B      

        if self.args.training.debug_mode:
            if torch.any(pix_idxs == -1):
                self.args.logger.error(f"some pixels are not sampled")
                return None
            
        return pix_idxs

    def _pixStrategyRandom(
        self,
        B:int,
    ):
        """
        Sample random pixels.
        Args:
            B: batch size; int
        Returns:
            pix_idxs: pixel indices of batch; torch.tensor (B,)
        """
        WH = self.img_wh[0]*self.img_wh[1]
        return torch.randint(0, WH, size=(B,), device=self.args.device, dtype=torch.int32)
    
    def _pixStrategyEntireImg(
        self,
    ):
        """
        Sample entire pixels.
        Returns:
            pix_idxs: pixel indices of batch; torch.tensor (B,)
        """
        WH = self.img_wh[0]*self.img_wh[1]
        return torch.arange(0, WH, device=self.args.device, dtype=torch.int32)
    
    def _pixStrategyClosest(
        self,
        img_idxs:torch.tensor
    ):
        """
        Sample pixel that is estimated to be closest by USS.
        Args:
            img_idxs: image indices of batch; torch.tensor (B,)
        Returns:
            pix_idxs: pixel indices of batch; torch.tensor (B,)
        """
        pix_idxs, _, _ = self.sensors_dict["USS"].getStatsForBatch(
            batch_img_idxs=img_idxs,
        )
        return pix_idxs
    
    def _pixStrategyValidDepth(
        self,
        B:int,
        sensor_type:str,
    ):
        """
        Sample random pixels with valid depth of particular sensor.
        Args:
            B: batch size; int
            sensor_type: type of sensor model, either 'USS' or 'ToF'; str
        Returns:
            pix_idxs: pixel indices of batch; torch.tensor (B,)
        """
        mask = self.sensors_dict[sensor_type].mask
        mask_idxs = torch.where(mask)[0]
        rand_ints = torch.randint(0, mask_idxs.shape[0], (B,), device=self.args.device, dtype=torch.int32)
        pix_idxs = mask_idxs[rand_ints]
        return pix_idxs
    
