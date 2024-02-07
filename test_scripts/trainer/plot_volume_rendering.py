import matplotlib.pyplot as plt
import numpy as np



def calcTrans(
    density:np.ndarray,
    delta:np.ndarray,
):
    """
    Calculate the transmittance of a sample
    Args:
        density: density of the sample; numpy array (N,)
        delta: delta of the sample; numpy array (N,)
    Returns:
        trans: transmittance of the sample; numpy array (N,)
    """
    trans = np.cumsum(density * delta)
    trans = np.exp(-trans)
    return trans

def calcWeight(
    density:np.ndarray,
    delta:np.ndarray,
):
    """
    Calculate the weight of a sample
    Args:
        density: density of the sample; numpy array (N,)
        delta: delta of the sample; numpy array (N,)
    Returns:
        weight: weight of the sample; numpy array (N,)
        trans: transmittance of the sample; numpy array (N,)
    """
    trans = calcTrans(
        density=density,
        delta=delta,
    )
    weight = 1 - np.exp(- density * delta)
    weight = trans * weight
    return weight, trans

def calcDepth(
    density:np.ndarray,
    delta:np.ndarray,
):
    """
    Calculate the depth of a sample
    Args:
        density: density of the sample; numpy array (N,)
        delta: delta of the sample; numpy array (N,)
    Returns:
        depth: final depth of the sample; float
        weight: weight of the sample; numpy array (N,)
        trans: transmittance of the sample; numpy array (N,)
    """
    weight, trans = calcWeight(
        density=density,
        delta=delta,
    )

    depth = np.cumsum(delta)
    depth = np.sum(weight * depth)
    return depth, weight, trans

def depthEstimation():
    num_sample = 500
    density_max = 2
    density_center = 1
    density_sigma = 0.3

    x = np.linspace(0, density_max, num_sample)
    delta = (density_max/num_sample) * np.ones_like(x)

    depths = []
    best_sig = None
    best_ampli = None
    # for sig in np.linspace(0.001, 10, 100):
    #     for ampli in np.linspace(0.001, 20, 100):
    #         density = ampli * np.exp(-(x - density_center)**2 / (2 * sig**2))

    #         depth, weight, trans = calcDepth(
    #             density=density,
    #             delta=delta,
    #         )
    #         depths.append(depth)

    #         if depth >= np.max(depths):
    #             best_sig = sig
    #             best_ampli = ampli

    # print(f"depths max: {np.max(depths)}")
    # print(f"best sig: {best_sig}, best ampli: {best_ampli}")

    
    # density = 5 * np.exp(-(x - density_center)**2 / (2 * density_sigma**2))
    density = 1.0 * np.ones_like(x)
    density[(x<0.5) | (x>1.5)] = 0.0

    

    depth, weight, trans = calcDepth(
        density=density,
        delta=delta,
    )

    # plot density, weight, transmittance
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    ax.plot(x, density, label='density')
    ax.plot(x, weight, label='weight')
    ax.plot(x, trans, label='transmittance')
    
    # ax.plot(x, np.cumsum(delta)*weight, label="depths")
    ax.vlines(depth, 0, 1, colors='r', linestyles='dashed', label='estimated depth')
    ax.legend()

    ax = axes[1]
    ax.plot(x, np.cumsum(delta)*weight, label="depths")
    ax.plot(x, weight/trans, label='weight_density')
    # ax.plot(x, np.cumsum(delta), label="depths")
    ax.legend()


    plt.show()

if __name__ == '__main__':
    depthEstimation()
