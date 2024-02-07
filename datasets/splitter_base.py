import numpy as np
import pandas as pd
from abc import abstractmethod

from args.args import Args


class Splitter():
    def __init__(
        self,
        args:Args,
        split_path:str,
        description_path:str,
    ) -> None:
        self.args = args
        self.split_path = split_path
        self.description_path = description_path

        self.rng = np.random.RandomState(
            seed=args.seed,
        )

    @abstractmethod
    def loadSplit(self):
        pass

    @abstractmethod
    def createSplit(self):
        pass

    @abstractmethod
    def getDatasetLengths(self):
        pass

    def splitDataset(
        self,
        split,
    ):
        dataset_lengths = self.getDatasetLengths()

        split_arrays = self.loadSplit(
            dataset_lengths=dataset_lengths,
        )
        if split_arrays is None:
            split_arrays = self.createSplit(
                dataset_lengths=dataset_lengths,
            )

            self.saveSplit(
                split_arrays=split_arrays,
            )

        split_masks = self.splitArrays2Masks(
            split=split,
            split_arrays=split_arrays,
            dataset_lengths=dataset_lengths,
        )
        return split_masks

    def saveSplit(
        self,
        split_arrays:dict,
    ):
        """
        Save the split to a csv file.
        Args:
            split_arrays: split of dataset {camera id: split array}; dict of bool array { str: bool array of shape (N_max) }
        """
        # save split and description
        pd.DataFrame(
            data=split_arrays,
            dtype=str,
        ).to_csv(
            path_or_buf=self.split_path,
            index=False,
        )

        split_ratio = self.args.dataset.split_ratio
        pd.DataFrame(
            data={
                'train':split_ratio['train'], 
                'val':split_ratio['val'], 
                'test':split_ratio['test'], 
                'keep_N_observations':str(self.args.dataset.keep_N_observations),
                'info':"This file contains the split ratios for this dataset. "
            },
            index=[0],
        ).to_csv(
            path_or_buf=self.description_path,
            index=False,
        )

    def splitArrays2Masks(
        self,
        split:str,
        split_arrays:np.array,
        dataset_lengths:dict,
    ):
        """
        Convert split array to mask.
        Args:
        split: 
            split to use; str
            split_arrays: split of dataset {camera id: split array}; dict of bool array { str: bool array of shape (N_max) }
            dataset_lengths: number of measurements per sensor stack {camera id: length}; dict of int { str: int }
        Returns:
            split_masks: split of dataset {camera id: split mask}; dict of bool array { str: bool array of shape (N_max) }
        """
        split_masks = {}
        for cam_id, split_array in split_arrays.items():
            # reduce split array from max size to actual size of this sensor stack
            split_array = split_array[:dataset_lengths[cam_id]] 
            split_masks[cam_id] = (split_array == split)
        return split_masks

    def getDatasetLengthsMax(
        self,
        dataset_lengths:dict,
    ):
        """
        Get the maximum length of the dataset.
        Args:
            dataset_lengths: lengths of measurements per sensor stack; dict of { cam_id: int }
        Returns:
            N_max: maximum length of dataset; int
        """
        N_max = 0
        for N in dataset_lengths.values():
            N_max = max(N_max, N)
        return N_max
    
    def createSkipArrays(
        self,
        dataset_lengths:dict,
    ):
        """
        Create skip arrays for each camera.
        Args:
            dataset_lengths: number of measurements per sensor stack {camera id: length}; dict of int { str: int }
        Returns:
            skip_arrays: skip arrays for each camera {camera id: skip array}; dict of bool array { str: bool array of shape (N_max) }
        """
        dataset_lengths_max = self.getDatasetLengthsMax(
            dataset_lengths=dataset_lengths,
        )

        skip_arrays = {}
        for cam_id, N in dataset_lengths.items():
            skip_array = N * ["skip"]
            skip_array += (dataset_lengths_max - N) * [np.nan]
            skip_arrays[cam_id] = np.array(skip_array, dtype='<U5')
        return skip_arrays



    