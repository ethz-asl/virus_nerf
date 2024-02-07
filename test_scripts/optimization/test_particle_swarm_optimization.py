import numpy as np
import os
import sys
 
sys.path.insert(0, os.getcwd())
from optimization.particle_swarm_optimization_wrapper import ParticleSwarmOptimizationWrapper
from optimization.metric import Metric
from optimization.plotter_metric import Plotter


def optimize(
    pso:ParticleSwarmOptimizationWrapper,
    metric:Metric,
    rand_termination:float=0.0,
):
    """
    Run optimization.
        N: number of particles; int
        M: number of hparams; int
        T: number of iterations; int
    Args:
        pso: particle swarm optimization; ParticleSwarmOptimizationWrapper
        metric: metric to optimize; Metric
        rand_termination: probability of random termination; float
    """
    
    terminate = pso._checkTermination()
    while not terminate:
        # get hparams to evaluate
        X = pso.getNextHparams(
            group_dict_layout=False,
            name_dict_layout=False,
        ) # np.array (M,)
        print(f"Iteration: t: {pso.t}, particle: {pso.n}, param: {X}")

        # evaluate metric
        score = metric(
            X=X,
        ) # float

        # update particle swarm
        terminate = pso.update(
            score=score,
        ) # bool

        # save state
        pso.saveState(
            score=score,
        )

        if rand_termination:
            rand_num = np.random.random()
            if rand_num < rand_termination:
                sys.exit()

def test_pso():
    # define optimization algorithm
    seeds = np.arange(9)
    T = 30 * 5
    termination_by_time = False
    hparams_lims_file = "test_scripts/optimization/hparams_lims.json"
    save_dirs = ["results/pso/test/opt"+str(i) for i in range(len(seeds))]
    metric_name = "rand"
    rand_termination = 0.0

    # plotter
    plotter = Plotter(
        num_axes=len(seeds),
    )

    for i, seed in enumerate(seeds):
        print(f"optimization {i+1}/{len(seeds)}")
        rng = np.random.default_rng(seed)

        # define particle swarm and metric
        pso = ParticleSwarmOptimizationWrapper(
            hparams_lims_file=hparams_lims_file,
            save_dir=save_dirs[i],
            T=T,
            termination_by_time=termination_by_time,
            rng=rng,
        )
        metric = Metric(
            metric_name=metric_name,
            hparams_lims=pso.hparams_lims,
            rng=rng,
            save_dir=save_dirs[i],
        )

        # run optimization
        optimize(
            pso=pso,
            metric=metric,
            rand_termination=rand_termination,
        ) # (N, T, M), (N, T)

        # plot results
        plotter.plot2D(
            pso=pso,
            metric=metric,
            ax_idx=i,
        )

    plotter.show(
        save_path="results/pso/test/opt.png",
    )

if __name__ == "__main__":
    test_pso()