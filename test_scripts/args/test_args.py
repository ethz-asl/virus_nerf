import os
import sys
 
sys.path.insert(0, os.getcwd())
from args.args import Args


def test_args():
    args = Args("hparams.json")
    args.saveJson()

if __name__ == "__main__":
    test_args()