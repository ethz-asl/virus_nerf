import torch
import os
import sys
import pandas as pd
from icecream import ic
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import matplotlib.ticker as mtick

sys.path.insert(0, os.getcwd())
from helpers.data_fcts import smoothIgnoreNans

colors = {
    'robot':    'red',
    'GT_map':   'grey', 
    'GT_scan':  'black',
    'NeRF':     'darkorange',
    'LiDAR':    'darkmagenta',
    'USS':      'blue',
    'ToF':      'green',
    'camera':   'lime',
}

def loadAblationStudy(
    base_dir,
    seeds,
):
    """
    Load ablation study results
    Args:
        base_dir: base directory; str
        seeds: seeds; list
    Returns:
        metrics_dict_list: list of dictionaries with results; list
    """
    metrics_dict_list = []
    for seed in seeds:
        log_file = os.path.join(base_dir, f"seed_{seed}", "logs.csv")
        df = pd.read_csv(log_file)

        time_full = df['time'].values
        step_full = df['step'].values
        mnn = df['mnn'].values
        psnr = df['psnr'].values

        valid = ~np.isnan(mnn)
        time = time_full[valid]
        step = step_full[valid]
        mnn = mnn[valid]
        psnr = psnr[valid]

        loss = df['loss'].values
        color_loss = df['color_loss'].values
        depth_loss = df['depth_loss'].values
        tof_loss = df['ToF_loss'].values
        uss_loss = df['USS_loss'].values

        metric_file = os.path.join(base_dir, f"seed_{seed}", "metrics.csv")
        df = pd.read_csv(metric_file, index_col=[0])
        zone_str =df.loc['NeRF', 'nn_mean']
        zone_str = zone_str.replace("'", '"')
        zone_dict = json.loads(zone_str)
        mnn_final = zone_dict['zone3']

        metrics_dict = {
            'time': time,
            'step': step,
            'mnn': mnn,
            'psnr': psnr,
            'mnn_final': mnn_final,
            'time_full': time_full,
            'step_full': step_full,
            'loss': loss,
            'color_loss': color_loss,
            'depth_loss': depth_loss,
            'tof_loss': tof_loss,
            'uss_loss': uss_loss,
        }

        metrics_dict_list.append(metrics_dict)
        
    # ic(metrics_dict_list)
    return metrics_dict_list

def plotMultipleLosses(
    colors:dict,
    metrics_dict_list:list,
    base_dir:str,
):
    len_max = np.max([len(metrics_dict['time_full']) for metrics_dict in metrics_dict_list])
    for metric in ['time_full', 'step_full', 'loss', 'color_loss', 'depth_loss', 'tof_loss', 'uss_loss']:
        for i, metrics_dict in enumerate(metrics_dict_list):
            values = np.full((len_max,), np.nan)
            values[:len(metrics_dict[metric])] = metrics_dict[metric]
            metrics_dict[metric] = values
            metrics_dict_list[i] = metrics_dict

            
    
    time_mean = np.nanmean([metrics_dict['time_full'] for metrics_dict in metrics_dict_list], axis=0)
    time_std = np.nanstd([metrics_dict['time_full'] for metrics_dict in metrics_dict_list], axis=0)
    step_mean = np.nanmean([metrics_dict['step_full'] for metrics_dict in metrics_dict_list], axis=0)
    step_std = np.nanstd([metrics_dict['step_full'] for metrics_dict in metrics_dict_list], axis=0)
    loss_mean = np.nanmean([metrics_dict['loss'] for metrics_dict in metrics_dict_list], axis=0)
    loss_std = np.nanstd([metrics_dict['loss'] for metrics_dict in metrics_dict_list], axis=0)
    color_loss_mean = np.nanmean([metrics_dict['color_loss'] for metrics_dict in metrics_dict_list], axis=0)
    color_loss_std = np.nanstd([metrics_dict['color_loss'] for metrics_dict in metrics_dict_list], axis=0)
    tof_loss_mean = np.nanmean([metrics_dict['tof_loss'] for metrics_dict in metrics_dict_list], axis=0)
    tof_loss_std = np.nanstd([metrics_dict['tof_loss'] for metrics_dict in metrics_dict_list], axis=0)
    uss_loss_mean = np.nanmean([metrics_dict['uss_loss'] for metrics_dict in metrics_dict_list], axis=0)
    uss_loss_std = np.nanstd([metrics_dict['uss_loss'] for metrics_dict in metrics_dict_list], axis=0)

    filter_size = 10
    loss_mean = smoothIgnoreNans(loss_mean, window_size=filter_size)
    color_loss_mean = smoothIgnoreNans(color_loss_mean, window_size=filter_size)
    tof_loss_mean = smoothIgnoreNans(tof_loss_mean, window_size=filter_size)
    uss_loss_mean = smoothIgnoreNans(uss_loss_mean, window_size=filter_size)

    if not np.allclose(step_std, 0):
        print("WARNING: step_std is not zero")
    
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))

    # plot loss
    ax = axes
    ax.plot(step_mean, loss_mean, c='black', label='total loss')   
    ax.plot(step_mean, color_loss_mean, c=colors['camera'], label='color loss')
    ax.fill_between(step_mean, color_loss_mean-color_loss_std, color_loss_mean+color_loss_std, alpha=0.2, color=colors['camera'])   
    ax.plot(step_mean, tof_loss_mean, c=colors['ToF'], label='ToF loss')
    ax.fill_between(step_mean, tof_loss_mean-tof_loss_std, tof_loss_mean+tof_loss_std, alpha=0.2, color=colors['ToF'])
    ax.plot(step_mean, uss_loss_mean, c=colors['USS'], label='USS loss')
    ax.fill_between(step_mean, uss_loss_mean-uss_loss_std, uss_loss_mean+uss_loss_std, alpha=0.2, color=colors['USS']) 

    ax.set_xlabel('step', color='black')
    ax.set_ylabel('loss')
    ax.set_ylim([0, 0.6])
    ax.set_xlim([step_mean[0], step_mean[-1]])

    ax3 = ax.twiny()
    color = 'black'
    ax3.set_xlabel('time [s]', color=color) 
    ax3.tick_params(axis='x', labelcolor=color)
    ax3.set_xlim([time_mean[0], time_mean[-1]])

    ax.legend(loc='upper right')

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(base_dir, "losses.png"))

def plotMultipleMetrics(
    colors:dict,
    metrics_dict_list:list,
    base_dir:str,
):
    len_max = np.max([len(metrics_dict['time']) for metrics_dict in metrics_dict_list])
    for metric in ['time', 'step', 'mnn', 'psnr']:
        for i, metrics_dict in enumerate(metrics_dict_list):
            values = np.full((len_max,), np.nan)
            values[:len(metrics_dict[metric])] = metrics_dict[metric]
            metrics_dict[metric] = values
            metrics_dict_list[i] = metrics_dict
    
    time_mean = np.nanmean([metrics_dict['time'] for metrics_dict in metrics_dict_list], axis=0)
    time_std = np.nanstd([metrics_dict['time'] for metrics_dict in metrics_dict_list], axis=0)
    step_mean = np.nanmean([metrics_dict['step'] for metrics_dict in metrics_dict_list], axis=0)
    step_std = np.nanstd([metrics_dict['step'] for metrics_dict in metrics_dict_list], axis=0)
    mnn_mean = np.nanmean([metrics_dict['mnn'] for metrics_dict in metrics_dict_list], axis=0)
    mnn_std = np.nanstd([metrics_dict['mnn'] for metrics_dict in metrics_dict_list], axis=0)
    psnr_mean = np.nanmean([metrics_dict['psnr'] for metrics_dict in metrics_dict_list], axis=0)
    psnr_std = np.nanstd([metrics_dict['psnr'] for metrics_dict in metrics_dict_list], axis=0)
    mnn_final_mean = np.nanmean([metrics_dict['mnn_final'] for metrics_dict in metrics_dict_list], axis=0)

    if not np.allclose(step_std, 0):
        print("WARNING: step_std is not zero")


    fig, axes = plt.subplots(1, 1, figsize=(5, 4))


    # plot mnn and psnr 
    ax = axes
    color = colors['NeRF']
    lns1 = ax.plot(step_mean, mnn_mean, c=color, label='NND')
    ar1 = ax.fill_between(step_mean, mnn_mean-mnn_std, mnn_mean+mnn_std, alpha=0.2, color=color, label='NND std')
    hln1 = ax.axhline(mnn_final_mean, linestyle="--", c=color, label='NND final')
    ax.set_xlabel('step', color='black') 
    ax.set_ylabel('Mean NND [m]')
    ax.set_ylim([0, 0.7])
    ax.yaxis.label.set_color(color) 
    ax.tick_params(axis='y', colors=color)
    ax.set_xlim([step_mean[0], step_mean[-1]])

    ax2 = ax.twinx()
    color = colors['camera']
    lns2 = ax2.plot(step_mean, psnr_mean, label='PSNR', c=color)
    ar2 = ax2.fill_between(step_mean, psnr_mean-psnr_std, psnr_mean+psnr_std, alpha=0.2, color=color, label='PSNR std')
    ax2.set_ylabel('PSNR [dB]')
    ax2.yaxis.label.set_color(color) 
    ax2.tick_params(axis='y', colors=color)
    ax2.set_ylim([0, 20])

    ax3 = ax.twiny()
    color = 'black'
    ax3.set_xlabel('time [s]', color=color) 
    ax3.tick_params(axis='x', labelcolor=color)
    ax3.set_xlim([time_mean[0], time_mean[-1]])

    lns = lns1 + lns2 + [hln1] #+ [ar1] + [ar2]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower right')
    # ax.set_title('Metrics')

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(base_dir, "training.png"))

def plot_online_training():
    base_dir = "results/ETHZ/online_training/commonroom"
    num_trainings = 10
    base_seed = 21
    seeds = np.arange(base_seed, base_seed+num_trainings)
    # seeds = [21, 22, 23 ,24, 26, 27, 28, 29]


    metrics_dict_list = loadAblationStudy(
        base_dir=base_dir,
        seeds=seeds,
    )

    plotMultipleMetrics(
        colors=colors,
        metrics_dict_list=metrics_dict_list,
        base_dir=base_dir,
    )

    plotMultipleLosses(
        colors=colors,
        metrics_dict_list=metrics_dict_list,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    plot_online_training()