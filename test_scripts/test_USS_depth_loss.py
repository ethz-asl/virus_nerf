import numpy as np
import matplotlib.pyplot as plt

def calcLosses(x, meas):
    depths_w = np.exp( -(x - np.min(x))/0.1 )
    # depths_w = depths_w / np.sum(depths_w)
    losses = np.abs(x - meas)
    depth_losses = depths_w * losses

    return losses, depths_w, depth_losses

def plotLosses(ax, x, losses, depths_w, depth_losses):
    ax.plot(x, losses, label="losses", color="blue")
    ax.scatter(x, depths_w, label="w", color="green")
    ax.scatter(x, depth_losses, label="result", color="red")
    ax.legend()

def main():
    num_points = 100
    fig, axis = plt.subplots(1, 1)
    meas = 0

    x = np.linspace(0, 1, num_points) + np.random.rand(num_points)/10
    losses, depths_w, depth_losses = calcLosses(x, meas)
    print(f"1, sum of depth losses: {np.sum(depth_losses)}")

    ax = axis
    plotLosses(ax=axis, x=x, losses=losses, depths_w=depths_w, depth_losses=depth_losses)
    ax.set_ylim([0, 1])


    plt.show()


def cos_loss():
    x = np.linspace(0.8, 1.5, 1000)
    meas = 1
    threshold = 0.2

    depth_error = x - meas
    cos_region_mask = (depth_error > -threshold/4) & (depth_error < threshold)
    lin_region_mask = depth_error <= -threshold/4

    y = np.zeros_like(x)
    y_cos = 0.5 * (1 - np.cos(2*np.pi * depth_error[cos_region_mask] / threshold))
    y_lin = 0.5 - np.pi/4 - depth_error[lin_region_mask] * np.pi / threshold 
    y[cos_region_mask] = y_cos
    y[lin_region_mask] = y_lin

    dy = np.zeros_like(x)
    dy_cos = np.sin(2*np.pi * depth_error[cos_region_mask] / threshold) * np.pi / threshold
    dy_lin = - np.pi / threshold
    dy[cos_region_mask] = dy_cos
    dy[lin_region_mask] = dy_lin

    plt.plot(x, y, label="loss")
    plt.plot(x, dy, label="gradient")
    plt.legend()
    plt.show()

def cos_lin_loss():
    x = np.linspace(-0.5, 0.5, 1000)
    meas = 0
    threshold = 0.25

    depth_error = x - meas
    cos_region_mask = (depth_error > -0.5*threshold) & (depth_error < threshold)
    lin_region_mask = depth_error <= -0.5*threshold

    y = (2*threshold/np.pi) * np.ones_like(x)
    y_cos = (threshold/np.pi) * (1 - np.cos(2*np.pi * depth_error[cos_region_mask] / (2*threshold)))
    y_lin = (2*threshold/np.pi) * (0.5 - np.pi/4 - depth_error[lin_region_mask] * np.pi / (2*threshold))
    y[cos_region_mask] = y_cos
    y[lin_region_mask] = y_lin

    dy = np.zeros_like(x)
    dy_cos = np.sin(2*np.pi * depth_error[cos_region_mask] / (2*threshold))
    dy_lin = - 1
    dy[cos_region_mask] = dy_cos
    dy[lin_region_mask] = dy_lin

    plt.ylim([-1.1, 1.1])
    plt.fill_between(x, -1.1, 1.1, where=x<0, color='green', alpha=0.2, label="inc. value")
    plt.fill_between(x, -1.1, 1.1, where=(x>0)&(x<threshold), color='brown', alpha=0.2, label="dec. value")
    plt.plot(x, y, label="loss")
    plt.plot(x, dy, label="gradient")
    plt.vlines(meas, -1.1, 1.1, color="red", label="measurement")
    plt.legend()
    plt.show()

def cos_quad_loss():
    x = np.linspace(-0.5, 0.5, 1000)
    meas = 0
    threshold = 0.25

    depth_error = x - meas
    cos_region_mask = (depth_error > -0.5*threshold) & (depth_error < threshold)
    quad_region_mask = depth_error <= -0.5*threshold

    y = (2*threshold/np.pi) * np.ones_like(x)
    y_cos = (threshold/np.pi) * (1 - np.cos(np.pi * depth_error[cos_region_mask] / threshold))
    y_quad = threshold * (1/np.pi - 0.25) + (1/threshold)*depth_error[quad_region_mask]**2
    y[cos_region_mask] = y_cos
    y[quad_region_mask] = y_quad

    dy = np.zeros_like(x)
    dy_cos = np.sin(2*np.pi * depth_error[cos_region_mask] / (2*threshold))
    dy_quad = (2/threshold) * depth_error[quad_region_mask]
    dy[cos_region_mask] = dy_cos
    dy[quad_region_mask] = dy_quad

    plt.ylim([-1.1, 1.1])
    plt.fill_between(x, -1.1, 1.1, where=x<0, color='green', alpha=0.2, label="inc. value")
    plt.fill_between(x, -1.1, 1.1, where=(x>0)&(x<threshold), color='brown', alpha=0.2, label="dec. value")
    plt.plot(x, y, label="loss")
    plt.plot(x, dy, label="gradient")
    plt.vlines(meas, -1.1, 1.1, color="red", label="measurement")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # main()
    # cos_lin_loss()
    cos_quad_loss()