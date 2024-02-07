import numpy as np
import matplotlib.pyplot as plt
import os
import sys
 
sys.path.insert(0, os.getcwd())
from helpers.plotting_fcts import combineImgs


def test_combineImgs():
    img1 = np.zeros((10, 10), dtype=bool)
    img1[0:5, 0:5] = True
    img2 = np.zeros((10, 10), dtype=bool)
    img2[3:10, 3:10] = True
    bool_imgs = [img2, img1]
    colors = ["red", "blue"]
    rgb_img = combineImgs(bool_imgs, colors)

    plt.imshow(rgb_img)
    plt.show()



if __name__ == "__main__":
    test_combineImgs()