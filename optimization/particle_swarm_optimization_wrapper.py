import numpy as np
import os
import json
import time
import pandas as pd

from optimization.particle_swarm_optimization import ParticleSwarmOptimization


class ParticleSwarmOptimizationWrapper(ParticleSwarmOptimization):
    def __init__(
        self,
        hparams_lims_file:str,
        save_dir:str,
        T:int,
        termination_by_time:bool,
        rng:np.random.Generator=None,
    ):
        """
        Initialize particle swarm optimization.
        Either T_iter or T_time must be defined.
        Args:
            hparams_lims_file: file path of json file containing hyper parameter limits; str
            save_dir: directory to save results to; str
            T_iter: maximum number of iterations or second depending on termination_by_time; int
            termination_by_time: whether to terminate after maximum time or maximum iterations elapsed; bool
            rng: random number generator; np.random.Generator
        """
        # general
        self.time_start = time.time() # start time
        self.time_offset = 0.0 # time offset to add from previous runs
        self.t = 0 # current iteration, [0, T_iter] where t=0 is the initial state
        self.T = T # maximum number of iterations or seconds depending on termination_by_time
        self.termination_by_time = termination_by_time # if true terminate after T seconds, else terminate after T iterations

        # read hparams limits
        self.hparams_lims, self.hparams_order, self.hparams_group = self._loadHparamsLimsFile(
            hparams_lims_file=hparams_lims_file,
        ) # np.array (M, 2), dict (key: str, value: int), dict (key: str, value: str)

        # PSO parameters
        pso_params_dict = {
            "num_dimensions": self.hparams_lims.shape[0],   # number of dimensions
            "num_particles": 5,        # number of particles
            "num_neighbours": 4,        # number of neighbours to consider for social component
            "alpha_momentum": 0.65,      # momentum coefficient
            "alpha_propre": 0.25,        # propre coefficient
            "alpha_social": 0.25,        # social coefficient
            "prob_explore": 0.75,       # probability to explore instead of exploit
        }

        # create or load state files
        self.pos_files, self.best_pos_files, self.vel_files, pso_init_dict, self.t, self.time_offset = self._createStateFiles(
            save_dir=save_dir,
            pso_params_dict=pso_params_dict,
        ) # list (N,), list (N,), list (N,), dict (key: str, value: np.arrays), int, float
            
        # initialize particle swarm optimization
        ParticleSwarmOptimization.__init__(
            self,
            rng=np.random.default_rng() if rng is None else rng,
            pso_params_dict=pso_params_dict,
            pso_init_dict=pso_init_dict,
            current_particle= self.t % pso_params_dict["num_particles"],
        )

        # save initial state if no state was loaded
        if self.t == 0:
            for i in range(pso_params_dict["num_particles"]):
                self.saveState(
                    score=np.inf,
                    particle=i,
                )

    def getNextHparams(
        self,
        group_dict_layout:bool=False,
        name_dict_layout:bool=False,
    ):
        """
        Get next hyper parameters of particle
        Args:
            group_dict_layout: whether to return hyper parameters as { group: { param: val } } or np.array (M,); bool
            name_dict_layout: whether to return hyper parameters as { param: val } or np.array (M,); bool
        Returns:
            hparams: hyper parameters; np.array (M,) or dict { group: { param: val } }
        """
        pos = self.getNextPos()

        if group_dict_layout:
            return self._pos2groupDict(
            pos=pos,
        )
        if name_dict_layout:
            return self._pos2nameDict(
            pos=pos,
        )
        return self._pos2hparam(
            pos=pos,
        )
    
    def update(
        self,
        score:float,
    ):
        """
        Update particle and check termination condition.
        Args:
            score: score of current particle; float
        Returns:
            terminate: whether to terminate optimization or not; bool
        """
        self.t += 1
        self.updateBestPos(
            score=score,
        )
        return self._checkTermination()
    
    def saveState(
        self,
        score:float,
        particle:int=None,
    ):
        """
        Save state of particle swarm optimization.
        Args:
            score: score of current particle; float
            particle: particle to save state of, if not given use current particle; int
        """
        i = particle
        if i is None:
            i = self.n

        # create name dictionaries
        name_dict = self._pos2nameDict(
            pos=self.pos[i],
        ) # dict (key: str, value: float)
        name_dict["score"] = score
        name_dict["time"] = (time.time() - self.time_start) + self.time_offset
        name_dict["iteration"] = self.t

        best_name_dict = self._pos2nameDict(
            pos=self.best_pos[i],
        ) # dict (key: str, value: float)
        best_name_dict["best_score"] = self.best_score[i]
        best_name_dict["best_count"] = self.best_count[i]

        vel_name_dict = self._hparam2nameDict(
            hparams=self.vel[i],
        ) # dict (key: str, value: float)

        # save updated csv file
        self._saveStateToFile(
            file_path=self.pos_files[i],
            name_dict=name_dict,
        )

        self._saveStateToFile(
            file_path=self.best_pos_files[i],
            name_dict=best_name_dict,
        )

        self._saveStateToFile(
            file_path=self.vel_files[i],
            name_dict=vel_name_dict,
        )

    def _loadState(
        self,
        save_dir:str,
        pso_params_dict:dict,
        pos_files:list,
        best_pos_files:list,
        vel_files:list,
    ):
        """
        Load state of particle swarm optimization.
        Args:
            save_dir: directory to load files from; str
            pso_params_dict: dictionary of PSO parameters; dict
            pos_files: list of file paths to position files; list (N,)
            best_pos_files: list of file paths to best position files; list (N,)
            vel_files: list of file paths to velocity files; list (N,)
        Returns:
            pso_init_dict: dictionary containing initial state of particle swarm optimization; dict (key: str, value: np.arrays)
            t: current iteration; int
            time: current time; float
        """
        N = pso_params_dict["num_particles"]
        M = pso_params_dict["num_dimensions"]

        # verify if PSO parameters are consistent
        pso_params_loaded = self._loadStateFromFile(
            file_path=os.path.join(save_dir, "pso_params.csv"),
        ) # dict (key: str, value: float)
        for key, value in pso_params_dict.items():
            if pso_params_loaded[key] != value:
                print(f"ERROR: ParticleSwarmOptimizationWrapper._loadState: PSO parameters are inconsistent."
                      + f" {key}: loaded={pso_params_loaded[key]} != given={value}")

        # load state from csv files
        pos = np.zeros((N, M))
        vel = np.zeros((N, M))
        best_pos = np.zeros((N, M))
        best_score = np.zeros((N,))
        best_count = np.zeros((N,), dtype=int)
        t = -1 # current iteration
        time = 0.0 # current time
        for i in range(N):
            # load state from csv file
            name_dict = self._loadStateFromFile(
                file_path=pos_files[i],
            ) # dict (key: str, value: float)
            if name_dict["iteration"] > t:
                t = name_dict["iteration"]
                time = name_dict["time"]
            del name_dict["score"]
            del name_dict["time"]
            del name_dict["iteration"]

            best_name_dict = self._loadStateFromFile(
                file_path=best_pos_files[i],
            ) # dict (key: str, value: float)
            best_score[i] = best_name_dict["best_score"]
            best_count[i] = best_name_dict["best_count"]
            del best_name_dict["best_score"]
            del best_name_dict["best_count"]

            vel_name_dict = self._loadStateFromFile(
                file_path=vel_files[i],
            ) # dict (key: str, value: float)

            # convert name_dict to pos
            pos[i] = self._nameDict2pos(
                name_dict=name_dict,
            ) # np.array (M,)

            best_pos[i] = self._nameDict2pos(
                name_dict=best_name_dict,
            ) # np.array (M,)

            vel[i] = self._nameDict2hparam(
                name_dict=vel_name_dict,
            ) # np.array (M,)  

        pso_init_dict = {
            "pos": pos,
            "vel": vel,
            "best_pos": best_pos,
            "best_score": best_score,
            "best_count": best_count,
        }
        return pso_init_dict, t, time

    def _saveStateToFile(
        self,
        file_path:str,
        name_dict:dict,
    ):
        """
        Save state of particle swarm optimization to file.
        Args:
            file_path: file path to save state to; str
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        """
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame(name_dict, index=[0])], axis=0, ignore_index=True)
        df.to_csv(file_path, index=False)

    def _loadStateFromFile(
        self,
        file_path:str,
        return_last_row:bool=True,
    ):
        """
        Load state of particle swarm optimization from file.
        Args:
            file_path: file path to load state from; str
            return_last_row: whether to return last row or all rows; bool
        Returns:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        """
        df = pd.read_csv(file_path)

        if return_last_row:
            return df.iloc[-1].to_dict()

        return df.reset_index().to_dict(orient="list")

    def _loadHparamsLimsFile(
        self,
        hparams_lims_file:str,
    ):
        """
        Read hyper parameter limits from json file
        Args:
            hparams_lims_file: file path of json file; str
        Returns:
            hparams_lims: hyper parameters limits; np.array (M, 2)
            hparams_order: hyper parameters order; dict (key: str, value: int)
            hparams_group: hyper parameters group; dict (key: str, value: str)
        """
        with open(hparams_lims_file) as f:
            group_dict = json.load(f)

        name_dict, hparams_group = self._groupDict2nameDict(
            group_dict=group_dict,
            return_groups=True,
        )

        hparams_order = {}
        hparams_lims = []
        for i, (param, lims) in enumerate(name_dict.items()):
            hparams_order[param] = i
            hparams_lims.append(lims)
        hparams_lims = np.array(hparams_lims)
        
        return hparams_lims, hparams_order, hparams_group
    
    def _createStateFiles(
        self,
        save_dir:str,
        pso_params_dict:dict,
    ):
        """
        Create state files.
        Args:
            save_dir: directory to save files to; str
            pso_params_dict: dictionary of PSO parameters; dict
        Returns:
            pos_files: list of file paths to position files; list (N,)
            best_pos_files: list of file paths to best position files; list (N,)
            vel_files: list of file paths to velocity files; list (N,)
            pso_init_dict: dictionary containing initial state of particle swarm optimization; dict (key: str, value: np.arrays)
            t: current iteration; int
            time: current time; float
        """
        N = pso_params_dict["num_particles"]

        # create save files
        pos_files = [os.path.join(save_dir, "pso_pos_"+str(i)+".csv") for i in range(N)]
        best_pos_files = [os.path.join(save_dir, "pso_best_pos_"+str(i)+".csv") for i in range(N)]
        vel_files = [os.path.join(save_dir, "pso_vel_"+str(i)+".csv") for i in range(N)]

        # load state if save directory already exists
        if os.path.exists(save_dir):
            pso_init_dict, t, time_offset = self._loadState(
                save_dir=save_dir,
                pso_params_dict=pso_params_dict,
                pos_files=pos_files,
                best_pos_files=best_pos_files,
                vel_files=vel_files,
            ) # dict (key: str, value: np.arrays), int, float  
            return pos_files, best_pos_files, vel_files, pso_init_dict, t, time_offset

        # create save directory and files
        os.makedirs(save_dir)
        name_list = [param for param in self.hparams_order.keys()]
        for i in range(N):
            pd.DataFrame(
                columns=name_list + ["score"]
            ).to_csv(pos_files[i], index=False)
            pd.DataFrame(
                columns=name_list + ["best_score", "best_count"]
            ).to_csv(best_pos_files[i], index=False)    
            pd.DataFrame(
                columns=name_list
            ).to_csv(vel_files[i], index=False) 
        pd.DataFrame(
            pso_params_dict, 
            index=[0]
        ).to_csv(os.path.join(save_dir, "pso_params.csv"), index=False)  

        # define initial values
        pso_init_dict = None
        t = 0
        time_offset = 0.0
        return pos_files, best_pos_files, vel_files, pso_init_dict, t, time_offset      

    def _checkTermination(
        self,
    ):
        """
        Check if optimization should be terminated.
        Returns:
            terminate: whether to terminate optimization; bool
        """
        if self.termination_by_time:
            if ((time.time() - self.time_start) + self.time_offset) >= self.T and (self.n == (self.N-1)):
                return True
        else:
            if (self.t >= self.T) and (self.n == (self.N-1)):
                return True
        return False
    
    def _groupDict2pos(
        self,
        group_dict:dict,
    ):
        """
        Convert hyper parameters { group: { param: val } } to position in particle space.
        Args:
            group_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        Returns:
            pos: particle space; np.array (M,)
        """
        name_dict = self._groupDict2nameDict(
            group_dict=group_dict,
        )
        pos = self._nameDict2pos(
            name_dict=name_dict,
        )
        return pos
    
    def _pos2groupDict(
        self,
        pos:np.array,
    ):
        """
        Convert position in particle space to hyper parameters { group: { param: val } }.
        Args:
            pos: particle space; np.array (M,)
        Returns:
            group_dict: dictionary containing hyper parameters; dict (key: str, value: str)
        """
        name_dict = self._pos2nameDict(
            pos=pos,
        )
        group_dict = self._nameDict2groupDict(
            name_dict=name_dict,
        )
        return group_dict
    
    def _nameDict2pos(
        self,
        name_dict:dict,
    ):
        """
        Convert hyper parameters { param: val } to position in particle space.
        Args:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        Returns:
            pos: particle space; np.array (M,)
        """
        hparams = self._nameDict2hparam(
            name_dict=name_dict,
        )
        pos = self._hparam2pos(
            hparams=hparams,
        )
        return pos
    
    def _pos2nameDict(
        self,
        pos:np.array,
    ):
        """
        Convert position in particle space to hyper parameters { param: val }.
        Args:
            pos: particle space; np.array (M,)
        Returns:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        """
        hparams = self._pos2hparam(
            pos=pos,
        )
        name_dict = self._hparam2nameDict(
            hparams=hparams,
        )
        return name_dict
    
    def _hparam2pos(
        self,
        hparams:np.array,
    ):
        """
        Convert hyper parameters to particle space.
        Args:
            hparams: hyper parameters; np.array (M,) or (N, M)
        Returns:
            pos: particle space; np.array (M,) or (N, M)
        """
        return (hparams - self.hparams_lims[:,0]) / (self.hparams_lims[:,1] - self.hparams_lims[:,0])
    
    def _pos2hparam(
        self,
        pos:np.array,
    ):
        """
        Convert particle space to hyper parameters.
        Args:
            pos: particle space; np.array (M,) or (N, M)
        Returns:
            hparams: hyper parameters; np.array (M,) or (N, M)
        """
        return pos * (self.hparams_lims[:,1] - self.hparams_lims[:,0]) + self.hparams_lims[:,0]
    
    def _nameDict2hparam(
        self,
        name_dict:dict,
    ):
        """
        Convert hyper-parameter dictionary from { param: val } to np.array (M,).
        Args:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float) 
                                                                or (key: str, value: list of floats)
        Returns:
            hparams: hyper parameters; np.array (M,) or (M, lenght of list)
        """
        hparams = len(self.hparams_order) * [None]
        for param, i in self.hparams_order.items():
            hparams[i] = name_dict[param]
        return np.array(hparams)
    
    def _hparam2nameDict(
        self,
        hparams:np.array,
    ):
        """
        Convert hyper-parameter dictionary from np.array (M,) to { param: val }.
        Args:
            hparams: hyper parameters; np.array (M,)   
        Returns:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        """
        name_dict = {}
        for param, i in self.hparams_order.items():
            name_dict[param] = hparams[i]
        return name_dict
    
    def _nameDict2groupDict(
        self,
        name_dict:dict,
    ):
        """
        Convert hyper-parameter dictionary from
            { param: val } to { group: { param: val } }
        Args:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        Returns:
            group_dict: dictionary containing hyper parameters; dict (key: str, value: str)
        """
        group_dict = { group: {} for group in np.unique(list(self.hparams_group.values())) }
        for param, val in name_dict.items():
            group_dict[self.hparams_group[param]][param] = val
        return group_dict
    
    def _groupDict2nameDict(
        self,
        group_dict:dict,
        return_groups:bool=False,
    ):
        """
        Convert hyper-parameter dictionary from
            { group: { param: val } } to { param: val }
        Args:
            group_dict: dictionary containing hyper parameters; dict (key: str, value: str)
            return_groups: whether to return groups or not; bool
        Returns:
            name_dict: dictionary containing hyper parameters; dict (key: str, value: float)
        """
        name_dict = {}
        groups = {}
        for group, group_params in group_dict.items():
            for param, val in group_params.items():
                if param in name_dict:
                    print(f"ERROR: ParticleSwarmOptimization._group2nameHparams: parameter {param} is defined multiple times.")
                name_dict[param] = val   
                groups[param] = group

        if return_groups:
            return name_dict, groups
        return name_dict