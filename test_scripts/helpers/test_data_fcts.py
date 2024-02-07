import numpy as np
import os
import sys
 
sys.path.insert(0, os.getcwd())
from helpers.data_fcts import linInterpolateArray, linInterpolateNans, dataConverged


def test_linInterpolateArray():
    x1 = np.array([1, 2, 3, 4, 5])
    y1 = 2*x1
    x2 = np.array([2.5, 4.5, 3.5, 4.5])

    y2 = linInterpolateArray(x1, y1, x2)
    print(f"y1: {y1}")
    print(f"y2: {y2}")

def test_linInterpolateNans():
    arr = np.array([np.nan, np.nan, 30, 40, 50])
    arr = linInterpolateNans(arr)
    print(f"arr: {arr}")

def test_dataConverged():
    arr = np.array([1, 2, 3, 4, 1])
    thr = 2.5
    idx = dataConverged(
        arr=arr,
        threshold=thr,
        data_increasing=True,
    )
    print(f"idx: {idx}")

    arr = np.array([5,3,2,1,0])
    thr = 1.5
    idx = dataConverged(
        arr=arr,
        threshold=thr,
        data_increasing=False,
    )
    print(f"idx: {idx}")


if __name__ == "__main__":
    # test_linInterpolateArray()
    # test_linInterpolateNans()
    test_dataConverged()