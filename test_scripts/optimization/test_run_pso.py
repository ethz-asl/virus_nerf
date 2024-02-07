import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import subprocess


def main():
    # run pso optimization
    cwd = os.getcwd()
    run_pso_path = os.path.join(cwd, "test_scripts", "optimization", "test_particle_swarm_optimization.py")

    while True:
        print("running pso")
        exit_code = subprocess.call(["python", run_pso_path])
        print("exit code:", exit_code)


if __name__ == "__main__":
    main()