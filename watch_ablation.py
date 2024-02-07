import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import subprocess
from helpers.system_fcts import checkGPUMemory


def main():
    # run pso optimization
    cwd = os.getcwd()
    run_ablation_path = os.path.join(cwd, "run_ablation.py")

    for _ in range(10):
        print("WATCHING OPTIMIZATION")

        checkGPUMemory()
        
        exit_code = subprocess.call(["python3", run_ablation_path])
        print("EXIT CODE:", exit_code)

if __name__ == "__main__":
    main()