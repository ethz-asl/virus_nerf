import numpy as np
import torch
import os
import sys
import gc
 
sys.path.insert(0, os.getcwd())
from helpers.system_fcts import get_size, moveToRecursively

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
class Parent():
    def __init__(self):
        self.model = Model()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.array = np.ones((1000,1000), dtype=np.float32)
        self.tensor = torch.ones((1000,1000), dtype=torch.float32)


def test_freeMemory():
    parent = Parent()
    
    print(f"References to trainer: {sys.getrefcount(parent)}")
    print(f"Size of trainer: {get_size(parent)}")

    moveToRecursively(
        obj=parent,
        destination="cpu",
    )
    del parent
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == "__main__":
    test_freeMemory()