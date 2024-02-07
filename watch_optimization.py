import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import subprocess
import nvidia_smi


def main():
    # run pso optimization
    cwd = os.getcwd()
    run_pso_path = os.path.join(cwd, "run_optimization.py")

    while True:
        print("running pso")

        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) # gpu id 0
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f"Run PSO: Free memory: {(info.free/1e6):.2f}Mb / {(info.total/1e6):.2f}Mb = {(info.free/info.total):.3f}%")
        nvidia_smi.nvmlShutdown()
        
        exit_code = subprocess.call(["python3", run_pso_path])
        print("exit code:", exit_code)

        # print("exit code:", exit_code)
        # if exit_code != 0:
        #     break

if __name__ == "__main__":
    main()