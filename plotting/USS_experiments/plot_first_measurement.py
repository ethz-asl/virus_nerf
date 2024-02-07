import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
import os
import sys
 
sys.path.insert(0, os.getcwd())
from helpers.uss_experiments_fcts import correctMeas, loadData, linInterpolate

def main():
    sensor = "MB1603" # either "URM37", "HC-SR04" or "MB1603"

    # get distances and angles
    dists = [0.25, 0.5, 1.0, 2.0]
    angles = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    objects = ['large', 'medium', 'small']
    surfaces = ['cardboard', 'plexiglas']

    # Create colormap
    cmap = plt.colormaps.get_cmap('plasma')
    # cNorm  = plt.Normalize(vmin=0, vmax=0.9)
    cNorm = LogNorm(vmin=0.01, vmax=1.0)

    # Create a figure and axis with polar projection
    fig, axis = plt.subplots(ncols=len(surfaces), nrows=len(objects), subplot_kw={'projection': 'polar'}, figsize=(10,9))
    fig.suptitle(sensor, fontsize=16, weight='bold')

    for k, object in enumerate(objects):
        for l, surface in enumerate(surfaces):
            ax = axis[k,l]

            # load dataframe
            df = loadData(sensor=sensor, object=object, surface=surface, measurement="first")
            df2 = loadData(sensor=sensor, object=object, surface=surface, measurement="second")

            # get mean, std and ratio for each distance and angle
            means = np.zeros((len(dists), len(angles)), dtype=float)
            stds = np.zeros((len(dists), len(angles)), dtype=float)
            ma_error = np.zeros((len(dists), len(angles)), dtype=float)
            rma_error = np.zeros((len(dists), len(angles)), dtype=float)
            for i, dist in enumerate(dists):
                # get mean of second measurement
                if f"{float(dist)}m_0deg" in df2.columns:
                    meas2 = df2[f"{float(dist)}m_0deg"].values
                elif f"{int(dist)}m_0deg" in df2.columns:
                    meas2 = df2[f"{int(dist)}m_0deg"].values
                meas2 = correctMeas(meas=meas2, first_meas=False)
                mean2 = np.mean(meas2)

                for j, angle in enumerate(angles):
                    if f"{dist}m_{int(angle)}deg" in df.columns:
                        meas = df[f"{dist}m_{int(angle)}deg"].values
                    elif f"{int(dist)}m_{int(angle)}deg" in df.columns:
                        meas = df[f"{int(dist)}m_{int(angle)}deg"].values
                    
                    meas = correctMeas(meas=meas, first_meas=True)  
                    meas = mean2 * (meas / np.mean(meas))

                    means[i,j] = np.mean(meas)
                    stds[i,j] = np.std(meas)
                    ma_error[i,j] = np.mean(np.abs(meas - dist))
                    rma_error[i,j] = np.mean(np.abs(meas - dist)) / dist

            a = np.deg2rad(linInterpolate(data=angles, check_for_invalid_data=False))
            for i, dist in enumerate(dists):
                m = linInterpolate(data=means[i])
                s = linInterpolate(data=stds[i])

                colours = cmap(cNorm(ma_error[i]))
                colours = np.concatenate((linInterpolate(data=colours[:,0]).reshape(-1,1), 
                                          linInterpolate(data=colours[:,1]).reshape(-1,1), 
                                          linInterpolate(data=colours[:,2]).reshape(-1,1),
                                          linInterpolate(data=colours[:,3]).reshape(-1,1)), axis=1)
                for j in range(len(a)-1):
                    # skip if measurement is not available
                    if m[j] == 0 or m[j+1] == 0:
                        continue

                    ax.plot(a[j:j+2], m[j:j+2], '-', color=colours[j])

                    vertices = [(a[j],m[j]-s[j]), 
                                (a[j],m[j]+s[j]), 
                                (a[j+1],m[j+1]+s[j+1]), 
                                (a[j+1],m[j+1]-s[j+1])]
                    ax.add_patch(
                        Polygon(vertices, closed=False, facecolor=colours[j], edgecolor=None, alpha=0.5)
                    )

            ax.set_theta_offset(np.pi / 2)  # Set the zero angle at the top
            ax.set_thetamin(-40)
            ax.set_thetamax(40)

            ax.set_xticks(np.deg2rad([-40, -20, 0, 20, 40]), labels=None) 
            ax.set_yticks([1.0, 2.0, 3.0], labels=None)
            ax.set_yticklabels(['1m', '2m', '3m'])           
            if k == 0:
                ax.set_xticklabels(['-40°', '-20°', '0°', '20°', '40°'])
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='both', color="grey", labelsize=11, labelcolor="black", pad=0.5)
            

            ax.set_thetagrids(angles=[-40, -30, -20, -10, 0, 10, 20, 30, 40], weight='black', alpha=0.5, labels=None)
            ax.set_rgrids(radii=[0.25, 0.5, 1.0, 2.0, 3.0], weight='black', alpha=0.5, labels=None)
            ax.set_ylim([0,3])

            if k == 0:
                ax.set_title(surface.capitalize(), weight='bold', y=1.05, fontsize=13)
            if l == 0:
                ax.set_ylabel(object.capitalize(), weight='bold', fontsize=13)     

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cNorm)
    sm.set_array(angles)
    cbar = plt.colorbar(sm, ax=axis.ravel().tolist()) 
    cbar.set_label('Mean Absolute Error [m]')  # Label for the colorbar
    plt.subplots_adjust(hspace=-0.15, wspace=-0.2, right=0.75, left=0)
    plt.savefig("plots/"+str(sensor)+".pdf")
    plt.savefig("plots/"+str(sensor)+".png")
    plt.show()


if __name__ == '__main__':
    main()