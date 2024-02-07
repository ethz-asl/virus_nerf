import os
import sys
 
sys.path.insert(0, os.getcwd())
from optimization.plotter_ethz import PlotterEthz



def main():
    data_dir = "results/pso/opt32_2"
    plotter = PlotterEthz(
        data_dir=data_dir,
    )
    plotter.plot()


if __name__ == "__main__":
    main()