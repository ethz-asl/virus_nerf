import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import time
from icecream import ic

from optimization.particle_swarm_optimization_wrapper import ParticleSwarmOptimizationWrapper
from args.args import Args
from datasets.dataset_rh import DatasetRH
from datasets.dataset_ethz import DatasetETHZ
from training.trainer import Trainer
from helpers.system_fcts import checkGPUMemory

def main():
    # define paraeters
    T = 36000 # if termination_by_time: T is time in seconds, else T is number of iterations
    termination_by_time = True # whether to terminate by time or iterations
    hparams_file = "ethz_usstof_gpu.json" 
    hparams_lims_file = "optimization/hparams_lims.json"
    save_dir = "results/pso/opt32_2"

    # get hyper-parameters and other variables
    args = Args(
        file_name=hparams_file
    )
    args.model.save = False
    args.training.debug_mode = False
    args.eval.eval_every_n_steps = args.training.max_steps + 1
    args.eval.plot_results = False
    args.eval.sensors = ["GT", "NeRF"]
    args.eval.num_color_pts = 0
    args.eval.batch_size = 8192
    args.training.batch_size = 4096
    args.seed = np.random.randint(0, 2**8-1)

    # datasets   
    if args.dataset.name == 'RH2':
        dataset = DatasetRH
    elif args.dataset.name == 'ETHZ':
        dataset = DatasetETHZ
    else:
        args.logger.error("Invalid dataset name.")    
    train_dataset = dataset(
        args = args,
        split="train",
    ).to(args.device)
    test_dataset = dataset(
        args = args,
        split='test',
        scene=train_dataset.scene,
    ).to(args.device)

    # pso
    pso = ParticleSwarmOptimizationWrapper(
        hparams_lims_file=hparams_lims_file,
        save_dir=save_dir,
        T=T,
        termination_by_time=termination_by_time,
        rng=np.random.default_rng(args.seed),
    )

    # run optimization
    terminate = False
    iter = 0
    while not terminate:
        iter += 1

        # get hparams to evaluate
        hparams_dict = pso.getNextHparams(
            group_dict_layout=True,
            name_dict_layout=False,
        ) # np.array (M,)

        # set hparams
        args.setRandomSeed(
            seed=args.seed+iter,
        )

        sampling_pix_sum = (hparams_dict["training"]["pixs_valid_uss"] + hparams_dict["training"]["pixs_valid_tof"])
        if sampling_pix_sum > 1.0:
            sampling_pix_sum = np.ceil(100*sampling_pix_sum) / 100 # round to 2 decimals
            hparams_dict["training"]["pixs_valid_uss"] /= sampling_pix_sum
            hparams_dict["training"]["pixs_valid_tof"] /= sampling_pix_sum
        sampling_strategy = {
            "imgs": "all", 
            "pixs": {
                "valid_uss": hparams_dict["training"]["pixs_valid_uss"],
                "valid_tof": hparams_dict["training"]["pixs_valid_tof"],
            },
        }
        for key, value in hparams_dict["training"].items():
            if (key == "pixs_valid_uss") or (key == "pixs_valid_tof"):
                setattr(args.training, "sampling_strategy", sampling_strategy)
                continue      
            setattr(args.training, key, value)

        for key, value in hparams_dict["occ_grid"].items():
            if (key == "update_interval") or (key == "decay_warmup_steps"):
                setattr(args.occ_grid, key, int(np.round(value)))
                continue 
            setattr(args.occ_grid, key, value)

        setattr(args.tof, "tof_pix_size", int(np.round(hparams_dict["ToF"]["tof_pix_size"])))

        print("\n\n----- NEW PARAMETERS -----")
        print(f"Time: {time.time()-pso.time_start+pso.time_offset:.1f}s, particle: {pso.n}")
        ic(hparams_dict)
        print(f"Current best mnn: {np.min(pso.best_score):.3f}, best particle: {np.argmin(pso.best_score)}")

        # load trainer
        trainer = Trainer(
            args=args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

        # train and evaluate model
        trainer.train()
        metrics_dict = trainer.evaluate()

        # get score
        score = metrics_dict['NeRF']["nn_mean"]['zone3']
        if score == np.nan:
            score = np.inf

        # update particle swarm
        terminate = pso.update(
            score=score,
        ) # bool

        # save state
        pso.saveState(
            score=score,
        )

        del trainer
        if checkGPUMemory():
            terminate = True


if __name__ == "__main__":
    main()