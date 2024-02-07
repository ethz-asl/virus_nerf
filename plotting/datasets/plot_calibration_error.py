import numpy as np
import matplotlib.pyplot as plt


def plot_calibration_error():
    error = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    mnn = [0.11384710895696395, 0.11123422475981287, 0.098474841, 0.103809233, 0.136416359, 0.176717544, 0.178960875, 0.230771927]
    convergence_50 = [29.774851322174072, 23.846983194351196, 64.06387997, 54.52338767, 51.33528614, 21.23815727, 21.24396801, 21.22101927]
    convergence_25 = [83.09322261810303, 50.12895464897156, 74.48167086, 68.98784113, 58.90571499, 27.11833215, 69.07346869, 69.75696516]
    convergence_10 = [110.25486445426941, 85.03390049934387, 77.9802444, 94.47088695, 105.5333767, 0.0, 118.6691425, 119.8912277]

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
    x_axis = np.arange(len(error))
    
    ax = axes[0]
    ax.bar(x_axis, mnn, color ='blue', width = 0.4)
    ax.set_xlabel("Angular error [degree]")
    ax.set_ylabel("Mean NND [m]")
    ax.set_xticks(x_axis, error)

    ax = axes[1]
    ax.bar(x_axis - 0.2, convergence_50, color ='blue', width = 0.2, label='50%')
    ax.bar(x_axis, convergence_25, color ='orange', width = 0.2, label='25%')
    ax.bar(x_axis + 0.2, convergence_10, color ='green', width = 0.2, label='10%')
    ax.set_xlabel("Angular error [degree]")
    ax.set_ylabel("Convergence time [s]")
    ax.legend()
    ax.set_xticks(x_axis, error)
    
    plt.savefig('test_scripts/datasets/tof_calibration_error.png')
    plt.show()

if __name__ == "__main__":
    plot_calibration_error()