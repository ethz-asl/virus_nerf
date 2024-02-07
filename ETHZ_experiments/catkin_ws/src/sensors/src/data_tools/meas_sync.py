#!/usr/bin/env python
import numpy as np
import os
from matplotlib import pyplot as plt
import copy

from rosbag_wrapper import RosbagWrapper



class MeasSync(RosbagWrapper):
    def __init__(
        self,
        data_dir:str,
        bag_name:str,
    ) -> None:
        self.data_dir = data_dir
        
        super().__init__(
            data_dir=data_dir,
            bag_name=bag_name,
        )
        
        self.plot_range = [0, 6]
        self.meas_error_max_uss = 10.0
        self.meas_error_max_tof = 10.0
        self.meas_error_max_depth = 10.0
        self.time_error_max_uss = 0.15
        self.time_error_max_tof = 0.05
        self.time_error_max_depth = 0.05
    
    def __call__(
        self,
        meass_dict:dict,
        times_dict:dict,
    ):
        """
        Synchronize time stamps of USS and ToF to RS time stamps.
        Args:
            meass_dict: dictionary with measurements; dict
            times_dict: dictionary with times; dict
        Returns:
            topics_read: topics to read from bag; list of str
            topics_write: topics to write to bag; list of str
            times_read: time stamps to read from bag; list of np.array of floats
            times_write: time stamps to write to bag; list of np.array of floats
            masks: masks of valid measurements; dict of np.array of bools
        """
        topics_read_1, topics_write_1, times_read_1, times_write_1, mask_1 = self._syncTime(
            stack_id=1,
            meass_dict=meass_dict,
            times_dict=times_dict,
        )
        topics_read_3, topics_write_3, times_read_3, times_write_3, mask_3 = self._syncTime(
            stack_id=3,
            meass_dict=meass_dict,
            times_dict=times_dict,
        )
        
        topics_read = topics_read_1 + topics_read_3
        topics_write = topics_write_1 + topics_write_3
        times_read = times_read_1 + times_read_3
        times_write = times_write_1 + times_write_3
        masks = {
            "CAM1": mask_1,
            "CAM3": mask_3,
        }
        return topics_read, topics_write, times_read, times_write, masks
    
    def _syncTime(
        self,
        stack_id:int,
        meass_dict:dict,
        times_dict:dict,
    ):
        """
        Synchronize time stamps of USS and ToF to RS time stamps.
        Args:
            stack_id: stack id; int
        Returns:
            topics_read: topics to read from bag; list of str
            topics_write: topics to write to bag; list of str
            times_read: time stamps to read from bag; list of np.array of floats
            times_write: time stamps to write to bag; list of np.array of floats
            mask: mask of valid measurements; np.array of bools
        """
        # load data not already done
        if "/USS"+str(stack_id) in times_dict.keys():
            times_uss = times_dict["/USS"+str(stack_id)]
        else:
            meass_dict, times_dict_uss = self.read(
                return_meas=[],
                return_time=["/USS"+str(stack_id)],
            )
            times_uss = times_dict_uss["/USS"+str(stack_id)]
            
        if "/TOF"+str(stack_id) in times_dict.keys():
            times_tof = times_dict["/TOF"+str(stack_id)]
        else:
            meass_dict, times_dict_tof = self.read(
                return_meas=[],
                return_time=["/TOF"+str(stack_id)],
            )
            times_tof = times_dict_tof["/TOF"+str(stack_id)]
            
        if "/CAM"+str(stack_id)+"/aligned_depth_to_color/image_raw" in times_dict.keys():
            times_depth = times_dict["/CAM"+str(stack_id)+"/aligned_depth_to_color/image_raw"]
        else:
            meass_dict, times_dict_depth = self.read(
                return_meas=[],
                return_time=["/CAM"+str(stack_id)+"/aligned_depth_to_color/image_raw"],
            )
            times_depth = times_dict_depth["/CAM"+str(stack_id)+"/aligned_depth_to_color/image_raw"]
            
        if "/CAM"+str(stack_id)+"/color/image_raw" in times_dict.keys():
            times_rs = times_dict["/CAM"+str(stack_id)+"/color/image_raw"]
        else:
            _, times_dict_rs = self.read(
                return_time=["/CAM"+str(stack_id)+"/color/image_raw"],
            )
            times_rs = times_dict_rs["/CAM"+str(stack_id)+"/color/image_raw"]
        
        # create figure for plotting
        fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 7))
        
        times_closest_uss, mask_uss, axs[0,0], axs[0,1] = self._findClosestTime(
            times_source=times_uss,
            times_target=times_rs,
            meass_source=None,
            ax_cor=axs[0,0],
            ax_err_time=axs[0,1],
            ax_err_meas=None,
            sensor="USS",
        )
        times_closest_tof, mask_tof, axs[1,0], axs[1,1] = self._findClosestTime(
            times_source=times_tof,
            times_target=times_rs,
            meass_source=None,
            ax_cor=axs[1,0],
            ax_err_time=axs[1,1],
            ax_err_meas=None,
            sensor="ToF",
        )
        times_closest_depth, mask_depth, axs[2,0], axs[2,1] = self._findClosestTime(
            times_source=times_depth,
            times_target=times_rs,
            meass_source=None,
            ax_cor=axs[2,0],
            ax_err_time=axs[2,1],
            ax_err_meas=None,
            sensor="DepthCam",
        )
        
        mask = mask_uss & mask_tof & mask_depth
        # mask = np.ones_like(mask_uss, dtype=np.bool_) # TODO: add mask
        
        topics_read = [
            "/USS"+str(stack_id), 
            "/TOF"+str(stack_id), 
            "/CAM"+str(stack_id)+"/aligned_depth_to_color/image_raw",
            "/CAM"+str(stack_id)+"/color/image_raw",
        ]
        topics_write = copy.copy(topics_read)
        times_read = [
            times_closest_uss[mask],
            times_closest_tof[mask],
            times_closest_depth[mask],
            times_rs[mask],
        ]
        times_write = [
            times_rs[mask],
            times_rs[mask],
            times_rs[mask],
            times_rs[mask],
        ]
        
        # plot results
        freq_rs = self._calcFreq(
            times=times_rs,
        )
        fig.suptitle(f"Synchronization on RS time stamps (RS freq = {freq_rs:.2f} Hz), keeping {mask.sum()}/{mask.shape[0]} samples")
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        path_splits = self.bag_path.split("/")
        plt.savefig(os.path.join(self.data_dir, path_splits[-1].replace(".bag", "") + f"_stack{stack_id}_sync.png"))
            
        return topics_read, topics_write, times_read, times_write, mask
    
    def _findClosestTime(
        self,
        sensor:str,
        times_source:np.array,
        times_target:np.array,
        meass_source:np.array,
        ax_cor:plt.Axes=None,
        ax_err_time:plt.Axes=None,
        ax_err_meas:plt.Axes=None,
    ):
        """
        Determine closest source measurement to target measurement:
        for every target time find closest source time
        Args:
            times_source: time stamps to adjust to times_target; np.array of floats (N,)
            times_target: reference time stamps; np.array of floats (M,)
            meass_source: measurements to adjust to times_target; np.array of floats (N,)
            sensor: sensor name; str
            ax_cor: axis to plot time correspondence; plt.Axes
            axs_err_time: axis to plot time error; plt.Axes
            axs_err_meas: axis to plot meas error; plt.Axes
        Returns:
            times_source_closest: corresponding time of source to time of target; np.array of ints (M,)
            mask: mask of valid measurements; np.array of bools (M,)
            ax_cor: axis; plt.Axes
            axs_err_time: axis; plt.Axes
            axs_err_meas: axis; plt.Axes
        """
        times_source = np.copy(times_source)
        times_target = np.copy(times_target)
        meass_source = np.copy(meass_source) if meass_source is not None else None
        
        times_source_rep = np.tile(times_source, (times_target.shape[0], 1)) # (M, N)
        times_target_rep = np.tile(times_target, (times_source.shape[0], 1)).T # (M, N)
        idxs_sort = np.argsort(np.abs(times_source_rep - times_target_rep), axis=1) # (M, N)
        idxs1 = idxs_sort[:,0] # (M,)
        idxs2 = idxs_sort[:,1] # (M,)
        
        times_source_closest = times_source[idxs1] # (M,)
        
        
        
        # create mask
        times_error = np.abs(times_target - times_source_closest) # (M,)
        if sensor == "USS":
            mask = times_error < self.time_error_max_uss
        elif sensor == "ToF":
            mask = times_error < self.time_error_max_tof
        elif sensor == "DepthCam":
            mask = times_error < self.time_error_max_depth
        
        if meass_source is not None:
            meass_error = np.abs(meass_source[idxs1] - meass_source[idxs2]) # (M,)
            
            # convert measurement errors to meters
            if sensor == "USS":
                meass_error = meass_error / 5000.0
            elif sensor == "ToF":
                meass_error = np.mean(meass_error, axis=1)
                meass_error = meass_error / 1000.0
            elif sensor == "DepthCam":
                meass_error = np.mean(meass_error, axis=1)
                meass_error = meass_error / 1000.0
            
            # create mask
            if sensor == "USS":
                mask = (meass_error < self.meas_error_max_uss) & mask
            elif sensor == "ToF":
                mask = (meass_error < self.meas_error_max_tof) & mask
            elif sensor == "DepthCam":
                mask = (meass_error < self.meas_error_max_depth) & mask
            
        
        if ax_cor is None:
            return times_source_closest, mask
        
        ax_cor = self._plotTimeCorrespondence(
            ax=ax_cor,
            times_source=times_source,
            times_target=times_target,
            mask=mask,
            sensor=sensor,
            idxs=idxs1,
        )
        
        if ax_err_time is None:
            return times_source_closest, mask, ax_cor
        
        ax_err_time = self._plotTimeError(
            ax=ax_err_time,
            times_target=times_target,
            times_error=times_error,
            sensor=sensor,
            mask=mask,
        )
        
        if (ax_err_meas is None) or (meass_source is None):
            return times_source_closest, mask, ax_cor, ax_err_time
        
        ax_err_meas = self._plotMeasError(
            ax=ax_err_meas,
            times_target=times_target,
            meass_error=meass_error,
            mask=mask,
            sensor=sensor,
        )
        return times_source_closest, mask, ax_cor, ax_err_time, ax_err_meas
    
    def _plotTimeCorrespondence(
        self,
        ax:plt.Axes,
        times_source:np.array,
        times_target:np.array,
        mask:np.array,
        sensor:str,
        idxs:np.array,
    ):        
        """
        Plot closest time.
        Args:
            ax: axis to plot on; plt.Axes
            times_source: time stamps to adjust to times_target; np.array of floats (N,)
            times_target: reference time stamps; np.array of floats (M,)
            mask: mask of valid measurements; np.array of bools (M,)
            sensor: sensor name; str
            idxs: indices of closest time; np.array of ints (M,)
        Returns:
            ax: axis with plot; plt.Axes
        """
        time_start = np.copy(times_target[0])
        times_source -= time_start
        times_target -= time_start
        
        colors_target, colors_source = self._getColors(
            M=times_target.shape[0],
            N=times_source.shape[0],
            idxs=idxs,
        )

        # determine which source sample is used for multiple target samples
        idxs_unique, idxs_inverse, idxs_counts = np.unique(idxs, return_inverse=True, return_counts=True)
        idxs_counts = idxs_counts[idxs_inverse] # (M,)
        mask_star_target = ~(idxs_counts > 1) # (M,)
        
        # convert star mask from target to source space
        mask_star = np.ones_like(times_source, dtype=np.bool_)
        mask_star[idxs] = mask_star_target
        
        ax.scatter(times_source[mask_star], 1.0 * np.ones_like(times_source[mask_star]), label="source", color=colors_source[mask_star])
        ax.scatter(times_source[~mask_star], 1.0 * np.ones_like(times_source[~mask_star]), label="source", color=colors_source[~mask_star], marker="s")
        ax.scatter(times_target[mask], 0.0*np.ones_like(times_target[mask]), label="times error", color=colors_target[mask])
        ax.scatter(times_target[~mask], 0.0*np.ones_like(times_target[~mask]), label="times error", facecolors="none", edgecolors=colors_target[~mask])
        
        ax.set_xlim(self.plot_range)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["RS", sensor])
        
        freq = self._calcFreq(
            times=times_source,
        )
        ax.set_title(f"Time correspondence ({sensor} freq = {freq:.2f} Hz)")
        return ax
    
    def _plotTimeError(
        self,
        ax:plt.Axes,
        times_target:np.array,
        times_error:np.array,
        sensor:str,
        mask:np.array,
    ):        
        """
        Plot closest time.
        Args:
            ax: axis to plot on; plt.Axes
            times_target: reference time stamps; np.array of floats (M,)
            times_error: time error; np.array of floats (M,)
            sensor: sensor name; str
            mask: mask of valid measurements; np.array of bools (M,)
        Returns:
            ax: axis with plot; plt.Axes
        """
        times_target = np.copy(times_target)
        times_error = np.copy(times_error)
        
        colors_target = self._getColors(
            M=times_target.shape[0],
        )
        
        ax.scatter(times_target[mask], times_error[mask], label="times error", color=colors_target[mask])
        ax.scatter(times_target[~mask], times_error[~mask], label="times error", facecolors="none", edgecolors=colors_target[~mask])

        if sensor == "USS":
            error_max = self.time_error_max_uss
        elif sensor == "ToF":
            error_max = self.time_error_max_tof
        elif sensor == "DepthCam":
            error_max = self.time_error_max_depth
        ax.hlines(error_max, self.plot_range[0], self.plot_range[1], label="max error", color="k", linestyle="--")

        ax.set_ylabel("error [s]")
        ax.set_xlim(self.plot_range)
        ax.set_title(f"Time error between {sensor} and RS")
        return ax
    
    def _plotMeasError(
        self,
        ax:plt.Axes,
        times_target:np.array,
        meass_error:np.array,
        mask:np.array,
        sensor:str,
    ):        
        """
        Plot closest time.
        Args:
            ax: axis to plot on; plt.Axes
            times_target: reference time stamps; np.array of floats (M,)
            meass_error: measurement error; np.array of floats (M,)
            mask: mask of valid measurements; np.array of bools (M,)
            sensor: sensor name; str
        Returns:
            ax: axis with plot; plt.Axes
        """
        meass_error = np.copy(meass_error)
        times_target = np.copy(times_target)
        
        colors_target = self._getColors(
            M=times_target.shape[0],
        )
        
        ax.scatter(times_target[mask], meass_error[mask], label="meas error", color=colors_target[mask])
        ax.scatter(times_target[~mask], meass_error[~mask], label="meas error", facecolors="none", edgecolors=colors_target[~mask])
        
        if sensor == "USS":
            error_max = self.meas_error_max_uss
        elif sensor == "ToF":
            error_max = self.meas_error_max_tof
        elif sensor == "DepthCam":
            error_max = self.meas_error_max_depth
        ax.hlines(error_max, self.plot_range[0], self.plot_range[1], label="max error", color="k", linestyle="--")
        
        ax.set_ylabel("error [m]")
        ax.set_xlabel("time [s]")
        ax.set_xlim(self.plot_range)
        ax.set_title(f"Depth error between 2 closest {sensor} samples")
        return ax
    
    def _getColors(
        self,
        M:int,
        N:int=None,
        idxs:np.array=None,
    ):
        """
        Get colors for plotting. If N is None, return only colors for target.
        Args:
            M: size of target; int
            N: size of source; int
            idxs: indices of closest time; np.array of ints (M,)
        Returns:
            colors_target: colors for target; list of str (M,)
            colors_source: colors for source; list of str (N,)
        """
        color_list = ["r", "g", "y", "c", "m", "b"]
        colors_target = np.array([color_list[i % len(color_list)] for i in range(M)])
        
        if N is None:
            return colors_target
        
        colors_source = np.array(["k" for _ in range(N)])
        colors_source[idxs] = colors_target 
        return colors_target, colors_source
        
    def _calcFreq(
        self,
        times:np.array,
    ):
        """
        Calculate average frequency of times.
        Args:
            times: time stamps; np.array of floats (N,)
        Returns:
            freq: average frequency of times; float
        """
        freq = 1.0 / np.mean(np.diff(times))
        return freq
        




def main():

    bag = RosbagSync(
        bag_path="/home/spadmin/catkin_ws_ngp/data/medium_2/medium_2_2.bag",
    )
    bag.syncBag(
        plot_dir="/home/spadmin/catkin_ws_ngp/data/medium_2/",
    )

if __name__ == "__main__":
    main()
    
    
    
    
 # def syncBag(
    #     self,
    #     plot_dir:str=None,
    # ):
        
    #     stack1_sync_paths = self.syncSensorStack(
    #         stack_id=1,
    #         plot_dir=plot_dir,
    #     )
    #     stack3_sync_paths = self.syncSensorStack(
    #         stack_id=3,
    #         plot_dir=plot_dir,
    #     )
    #     bag_sync_path = stack1_sync_paths + stack3_sync_paths + [self.bag_path]
    #     keep_topics = (len(stack1_sync_paths) + len(stack3_sync_paths)) * ["all"] \
    #                     + [["/CAM1/depth/color/points", "/CAM3/depth/color/points", "/rslidar_points"]]
    #     delete_ins = (len(stack1_sync_paths) + len(stack3_sync_paths)) * [True] + [False]
        
    #     self.merge(
    #         bag_path_out=self.bag_path.replace(".bag", "_sync.bag"),
    #         bag_paths_ins=bag_sync_path,
    #         keep_topics=keep_topics,
    #         delete_ins=delete_ins,
    #     )
    
    # def syncSensorStack(
    #     self,
    #     stack_id:int,
    #     plot_dir:str=None,
    # ):
    #     meass_uss, times_uss = self.read(
    #         topic="/USS"+str(stack_id),
    #     )
    #     meass_tof, times_tof = self.read(
    #         topic="/TOF"+str(stack_id),
    #     )
    #     _, times_rs = self.read(
    #         topic="/CAM"+str(stack_id)+"/color/image_raw",
    #     )
        
    #     axs = [None, None]
    #     if plot_dir is not None:
    #         fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 7))
        
    #     times_closest_uss, mask_uss, axs[0,0], axs[1,0], axs[2,0] = self._findClosestTime(
    #         times_source=times_uss,
    #         times_target=times_rs,
    #         meass_source=meass_uss,
    #         ax_cor=axs[0,0],
    #         ax_err_time=axs[1,0],
    #         ax_err_meas=axs[2,0],
    #         sensor="USS",
    #     )
    #     times_closest_tof, mask_tof, axs[0,1], axs[1,1], axs[2,1] = self._findClosestTime(
    #         times_source=times_tof,
    #         times_target=times_rs,
    #         meass_source=meass_tof,
    #         ax_cor=axs[0,1],
    #         ax_err_time=axs[1,1],
    #         ax_err_meas=axs[2,1],
    #         sensor="ToF",
    #     )
        
    #     mask = mask_uss & mask_tof
        
    #     bag_path_sync_uss = self.bag_path.replace(".bag", "_sync_uss"+str(stack_id)+".bag")
    #     self.writeTimeStamp(
    #         bag_path_sync=bag_path_sync_uss,
    #         topic_async="/USS"+str(stack_id),
    #         topic_sync="/USS"+str(stack_id),
    #         time_async=times_closest_uss[mask],
    #         time_sync=times_rs[mask],
    #     )
        
    #     bag_path_sync_tof = self.bag_path.replace(".bag", "_sync_tof"+str(stack_id)+".bag")
    #     self.writeTimeStamp(
    #         bag_path_sync=bag_path_sync_tof,
    #         topic_async="/TOF"+str(stack_id),
    #         topic_sync="/TOF"+str(stack_id),
    #         time_async=times_closest_tof[mask],
    #         time_sync=times_rs[mask],
    #     )
        
    #     bag_path_sync_cam = self.bag_path.replace(".bag", "_sync_rs"+str(stack_id)+".bag")
    #     self.writeTimeStamp(
    #         bag_path_sync=bag_path_sync_cam,
    #         topic_async="/CAM"+str(stack_id)+"/color/image_raw",
    #         topic_sync="/CAM"+str(stack_id)+"/color/image_raw",
    #         time_async=times_rs[mask],
    #         time_sync=times_rs[mask],
    #     )
    #     bag_path_sync_list = [bag_path_sync_uss, bag_path_sync_tof, bag_path_sync_cam]
        
        # if plot_dir is not None:
        #     freq_rs = self._calcFreq(
        #         times=times_rs,
        #     )
        #     fig.suptitle(f"Synchronization on RS time stamps (RS freq = {freq_rs:.2f} Hz), keeping {mask.sum()}/{mask.shape[0]} samples")
            
        #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        #     path_splits = self.bag_path.split("/")
        #     plt.savefig(os.path.join(plot_dir, path_splits[-1].replace(".bag", "") + f"_stack{stack_id}_sync.png"))
            
        # return bag_path_sync_list