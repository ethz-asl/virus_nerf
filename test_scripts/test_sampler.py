import numpy as np
import os
import sys
import matplotlib.pyplot as plt
 
sys.path.insert(0, os.getcwd())
from args.args import Args
from training.sampler import Sampler



def main():
    img_wh=(240, 320)

    # load hparams
    hparams_file = "rh_windows.json"
    args = Args(file_name=hparams_file)

    # create sampler
    sampler = Sampler(
        args=args,
        dataset_len=100,
        img_wh=img_wh,
        seed=0,
        sensors_dict=None,
    )

    # get pixel weights
    weights = sampler.weights.reshape(img_wh[1], img_wh[0])
    print(f"pixel weights shape: {weights.shape}")

    # plot weights
    plt.imshow(weights)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()