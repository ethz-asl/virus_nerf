import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys
 
sys.path.insert(0, os.getcwd())
from helpers.uss_experiments_fcts import correctMeas, loadData


def main():

    # get distances and angles
    dists = [0.25, 0.5, 1.0, 2.0]
    angle = 0
    objects = ['large', 'medium', 'small']
    surfaces = ['cardboard', 'plexiglas']
    sensors = ['HC-SR04', 'URM37', 'MB1603']

    # Create colormap
    cmap = plt.colormaps.get_cmap('plasma')
    # cNorm  = plt.Normalize(vmin=0, vmax=0.3)
    cNorm = LogNorm(vmin=0.01, vmax=1.0)

    # Create a figure and axis with polar projection
    fig, axis = plt.subplots(ncols=len(surfaces), nrows=len(objects), figsize=(9,9))

    for s, sensor in enumerate(sensors):

        for k, object in enumerate(objects):
            for l, surface in enumerate(surfaces):
                ax = axis[k,l]

                # load dataframe
                df = loadData(sensor=sensor, object=object, surface=surface, measurement="second")

                # get mean, std and ratio for each distance and angle
                means = np.zeros((len(dists)), dtype=float)
                stds = np.zeros((len(dists)), dtype=float)
                ma_error = np.zeros((len(dists)), dtype=float)
                rma_error = np.zeros((len(dists)), dtype=float)
                for i, dist in enumerate(dists):
                    if f"{dist}m_{int(angle)}deg" in df.columns:
                        meas = df[f"{dist}m_{int(angle)}deg"].values
                    elif f"{int(dist)}m_{int(angle)}deg" in df.columns:
                        meas = df[f"{int(dist)}m_{int(angle)}deg"].values

                    meas = correctMeas(meas=meas, first_meas=False)

                    means[i] = np.mean(meas)
                    stds[i] = np.std(meas)
                    ma_error[i] = np.mean(np.abs(meas - dist))
                    rma_error[i] = np.mean(np.abs(meas - dist)) / dist
                
                for i, dist in enumerate(dists):    
                    ax.scatter(s, means[i], s=30, color=cmap(cNorm(ma_error[i])))
                    ax.errorbar(s, means[i], yerr=stds[i], fmt='none', ecolor=cmap(cNorm(ma_error[i])), capsize=3, capthick=1)

                ax.set_yticks([0.25, 0.5, 1.0, 2.0], labels=None)
                if l == 0:
                    ax.set_yticklabels(['0.25m', '0.5m', '1m', '2m'])
                else:
                    ax.set_yticklabels([])

                ax.set_xticks([0, 1, 2], labels=None)
                if k == len(objects)-1:
                    ax.set_xticklabels(sensors)
                else:
                    ax.set_xticklabels([])

                ax.grid(axis='y', linewidth=0.5)
                ax.set_xlim([-0.3, 2.3])
                ax.set_ylim([0,2.25])

                if k == 0:
                    ax.set_title(surface.capitalize(), weight='bold', y=1.05, fontsize=12)
                if l == 0:
                    ax.set_ylabel(object.capitalize(), weight='bold', fontsize=12)

    # ax.legend(loc='upper right')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cNorm)
    sm.set_array([0,1,2])
    cbar = plt.colorbar(sm, ax=axis.ravel().tolist()) 
    cbar.set_label('Mean Absolute Error [m]')  # Label for the colorbar

    plt.subplots_adjust(hspace=0.1, wspace=0.08, right=0.75)
    plt.savefig("plots/all_sensors.pdf")
    plt.savefig("plots/all_sensors.png")
    plt.show()




if __name__ == '__main__':
    main()