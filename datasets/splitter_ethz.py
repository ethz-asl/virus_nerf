import numpy as np
import pandas as pd
import os
import sys

from datasets.splitter_base import Splitter
from args.args import Args
from helpers.data_fcts import sensorName2ID

class SplitterETHZ(Splitter):
    def __init__(
        self,
        args:Args,
    ):
        if not os.path.exists(os.path.join(args.ethz.dataset_dir, args.ethz.room, 'split')):
            os.mkdir(os.path.join(args.ethz.dataset_dir, args.ethz.room, 'split'))
    
        super().__init__(
            args=args,
            description_path = os.path.join(args.ethz.dataset_dir, args.ethz.room, 'split', 'split_description.csv'),
            split_path = os.path.join(args.ethz.dataset_dir, args.ethz.room, 'split', 'split.csv'),
        )

        self.time_thr = 0.1 # time threshold for matching times

    def getDatasetLengths(
        self,
    ):
        """
        Verify that the dataset length is the same for all sensors of one sensor stack.
        Returns:
            dataset_lengths: lengths of measurements per sensor stack; dict of { cam_id: int }
            dataset_lengths_max: maximum length of dataset; int
        """
        data_dir = os.path.join(self.args.ethz.dataset_dir, self.args.ethz.room)

        dataset_lengths = {}
        for cam_id in self.args.ethz.cam_ids:
            N = None # length of dataset
            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )
            df_names = [
                'measurements/USS'+str(id)+'.csv',
                'measurements/TOF'+str(id)+'.csv',
            ]
            for name in df_names:
                df = pd.read_csv(
                    filepath_or_buffer=os.path.join(data_dir, name),
                    dtype={'time':str, 'meas':np.float32},
                )
                if N is None:
                    N = df.shape[0]
                elif N != df.shape[0]:
                    self.args.logger.error(f"DatasetETHZ::_verifyDatasetLength: dataset length "
                                        + f"is not the same for all sensors!")
                    sys.exit()
                    
            dir_names = [
                'measurements/CAM'+str(id)+'_color_image_raw',
                'measurements/CAM'+str(id)+'_aligned_depth_to_color_image_raw',
            ]
            for name in dir_names:
                files = os.listdir(os.path.join(data_dir, name))
                if N != len(files):
                    self.args.logger.error(f"DatasetETHZ::_verifyDatasetLength: dataset length "
                                        + f"is not the same for all sensors!")
                    sys.exit()
            dataset_lengths[cam_id] = N

        return dataset_lengths
    
    def loadSplit(
        self,
        dataset_lengths:dict,
    ):
        """
        Load split if it already exists.
        Args:
            dataset_lengths: number of measurements per sensor stack; dict of { cam_id: int }
        Returns:
            split_masks: split of datasets; None or {str: bool array of shape (N_stack_i,)}
                        {camera id: bool array of shape (N_stack_i,)} if valid split exists
                        None if no valid split exists
        """
        # load split if it exists already
        df_description = None
        if (not os.path.exists(self.description_path)) or (not os.path.exists(self.split_path)):
            return None
        
        df_description = pd.read_csv(
            filepath_or_buffer=self.description_path,
            dtype={'info':str,'train':float, 'val':float, 'test':float, 'keep_N_observations':str},
        )
        df_split = pd.read_csv(
            filepath_or_buffer=self.split_path,
            dtype={'CAM1':str, 'CAM3':str},
        )
        
        # split ratio must be the same as in description (last split)
        split_ratio = self.args.dataset.split_ratio
        if (df_description['train'].values[0]!=split_ratio['train']) \
            or (df_description['val'].values[0]!=split_ratio['val']) \
            or (df_description['test'].values[0]!=split_ratio['test']) \
            or (df_description['keep_N_observations'].values[0] != str(self.args.dataset.keep_N_observations)):
            return None         

        # verify that split has same length as dataset
        for cam_id, N in dataset_lengths.items():
            split_array = df_split[cam_id].values
            split_array = split_array[:dataset_lengths[cam_id]]
            if np.any((split_array!='train') & (split_array!='val') & (split_array!='test') & (split_array!='skip')):
                return None

        # create split masks
        print(f"Loding split from {self.split_path}")
        split_arrays = {}
        for cam_id in dataset_lengths.keys():
            split_arrays[cam_id] = df_split[cam_id].values
        return split_arrays
    
    def createSplit(
        self,
        dataset_lengths:dict,
    ):
        """
        Create split of dataset.
        Args:
            dataset_lengths: number of measurements per sensor stack {camera id: length}; dict of int { str: int }
        Returns:
            split_arrays: split of dataset {camera id: split array}; dict of bool array { str: bool array of shape (N_max) }
        """
        times = self.loadTimes()
        common_idxs = self.matchTimes(
            times=times,
        )

        split_arrays = self.createSkipArrays(
            dataset_lengths=dataset_lengths,
        ) 
        common_array = self.createSplitArray(
            length=list(common_idxs.values())[0].shape[0], # length of common times
        )

        
        for cam_id in split_arrays.keys():
            split_arrays[cam_id][common_idxs[cam_id]] = common_array

        print(f"Creating new split at {self.split_path}")
        return split_arrays
    
    def createSplitArray(
        self,
        length:int,
    ):
        """
        Create split array for dataset.
        Args:
            length: length of dataset; int
        Returns:
            split_array: split of dataset; array of shape (length,)
        """
        # verify that split ratio is correct
        split_ratio = self.args.dataset.split_ratio
        if split_ratio['train'] + split_ratio['val'] + split_ratio['test'] != 1.0:
            self.args.logger.error(f"split ratios do not sum up to 1.0")
            sys.exit()

        # keep subset of dataset for testing
        N = length
        if self.args.dataset.keep_N_observations != 'all':
            N = self.args.dataset.keep_N_observations
            if N > length:
                self.args.logger.error(f"keep_N_observations is larger than dataset length")
                sys.exit()

        # create new split
        N_train = int(split_ratio['train']*N)
        N_val = int(split_ratio['val']*N)
        N_test = int(split_ratio['test']*N)

        rand_idxs = self.rng.permutation(length)
        train_idxs = rand_idxs[:N_train]
        val_idxs = rand_idxs[N_train:N_train+N_val]
        test_idxs = rand_idxs[N_train+N_val:N_train+N_val+N_test]

        split_array = np.array(length * ["skip"], dtype='<U5')
        split_array[train_idxs] = "train"
        split_array[val_idxs] = "val"
        split_array[test_idxs] = "test"
        return split_array
    
    def loadTimes(
        self,
    ):
        """
        Load times of measurements per sensor stack.
        Returns:
            times: times of measurements per sensor stack; dict of { cam_id: np.array of shape (N,) }
        """
        data_dir = os.path.join(self.args.ethz.dataset_dir, self.args.ethz.room)

        times = {}
        for cam_id in self.args.ethz.cam_ids:
            id = sensorName2ID(
                sensor_name=cam_id,
                dataset=self.args.dataset.name,
            )
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(data_dir, 'measurements/USS'+str(id)+'.csv'),
                dtype={'time':np.float64, 'meas':np.float32},
            )
            times[cam_id] = np.array(df['time'].values, dtype=np.float64)

        return times
    
    def matchTimes(
        self,
        times:dict,
    ):
        """
        Determine common times for all sensors and return indices of common times.
        Args:
            times: times of measurements per sensor stack; dict of { cam_id: np.array of shape (Ni,) }
        Returns:
            common_idxs: indices of common times; dict of { cam_id: np.array of shape (N_common,) }
        """
        cam_ids = self.args.ethz.cam_ids
        
        # find common times
        common_time = times[cam_ids[0]] # (N_common,)
        for cam_id in cam_ids[1:]:
            t1, t2 = np.meshgrid(common_time, times[cam_id], indexing='ij') # (N_common, Ni), (N_common, Ni)
            mask = np.abs(t1-t2) < self.time_thr

            # verify that each time matches at most one time
            if (np.any(np.sum(mask, axis=1) > 1)) or (np.any(np.sum(mask, axis=0) > 1)):
                self.args.logger.error(f"DatasetETHZ::matchTimes: at least one time stamp can be assigned to multiple times!")
                self.args.logger.error(f"mask_sums: {np.sum(mask, axis=1)}, {np.sum(mask, axis=0)}")
                sys.exit()

            common_time = common_time[np.sum(mask, axis=1) == 1] # (N_common,)
        
        # find common indices
        common_idxs = {}
        for cam_id in cam_ids:
            t1, t2 = np.meshgrid(common_time, times[cam_id], indexing='ij') # (N_common, Ni), (N_common, Ni)
            mask = np.abs(t1-t2) < self.time_thr # (N_common, Ni)
            common_idxs[cam_id] = np.where(mask)[1] # (N_common,)

            if self.args.training.debug_mode:
                if len(common_idxs[cam_id]) != len(common_time):
                    self.args.logger.error(f"DatasetETHZ::matchTimes: length of time is in common not consistent!")
                    self.args.logger.error(f"len(common_idxs[cam_id]): {len(common_idxs[cam_id])}, len(common_time): {len(common_time)}")
                    sys.exit()
        
        return common_idxs
