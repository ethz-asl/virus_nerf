import numpy as np


class ParticleSwarmOptimization():
    def __init__(
        self,
        rng:np.random.Generator,
        pso_params_dict:dict,
        pso_init_dict:dict=None,
        current_particle:int=0,
    ):
        """
        Initialize particle swarm optimization.
        Args:
            rng: random number generator; np.random.Generator
            pso_params_dict: dictionary of PSO parameters; dict
            pso_init_dict: dictionary of PSO initialization; dict
            current_particle: current particle; int
        """
        # general
        self.rng = rng

        # PSO parameters
        self.n = int(current_particle) # current particle, [0, N-1]
        self.N = pso_params_dict["num_particles"] # number of particles
        self.M = pso_params_dict["num_dimensions"] # number of dimensions
        self.num_neighbours = pso_params_dict["num_neighbours"] # number of neighbours
        self.alpha_momentum = pso_params_dict["alpha_momentum"] # momentum coefficient
        self.alpha_propre = pso_params_dict["alpha_propre"] # propre coefficient
        self.alpha_social = pso_params_dict["alpha_social"] # social coefficient
        self.prob_explore = pso_params_dict["prob_explore"] # probability of exploration
        self.exploring = True # whether to explore or exploit

        # initialize position and velocity of particles
        if pso_init_dict is None:
            self.pos, self.vel, self.best_pos, self.best_score, self.best_count = self._initParticles()
        else:
            self.pos = pso_init_dict["pos"]
            self.vel = pso_init_dict["vel"]
            self.best_pos = pso_init_dict["best_pos"]
            self.best_score = pso_init_dict["best_score"]
            self.best_count = pso_init_dict["best_count"]

        # decrease particle iterator n because it is increased in the beginning of the first iteration
        self._decreaseIterator()

    def getNextPos(
        self,
    ):
        """
        Update particle and get next position of particle.
        Returns:
            pos: position of particle; np.array (M,)
        """
        # update iterator n
        self._increaseIterator()

        # update particle
        self._updateParticle(
            n=self.n,
        )

        # explore new position or exploit best position
        prob = self.rng.random()
        if (self.best_count[self.n] == 0) or (prob < self.prob_explore):
            self.exploring = True
            return self.pos[self.n]
  
        self.exploring = False
        return self.best_pos[self.n]

    def updateBestPos(
        self,
        score:float,
    ):
        """
        Update best position of particle.
        Args:
            score: score of particle; float
        """
        n = self.n

        # update best score of particle in case of exploration or exploitation
        if self.exploring: 
            if score < self.best_score[n]:
                self.best_score[n] = score
                self.best_pos[n] = self.pos[n]
                self.best_count[n] = 1
        else:
            self.best_score[n] = (self.best_count[n]*self.best_score[n] + score) / (self.best_count[n] + 1)
            self.best_count[n] += 1

    def _initParticles(
        self,
    ):
        """
        Initialize particles.
        Returns:
            pos: particle space; np.array (N, M)
            vel: particle velocity; np.array (N, M)
            best_pos: best position of particle; np.array (N, M)
            best_score: best score of particle; np.array (N,)
            best_count: number of times best score was updated; np.array (N,)
        """
        pos = self._initPosRandom() # (N, M)
        vel = self._initVelRandom() # (N, M)
        best_pos = np.zeros_like(pos) # (N, M)
        best_score = np.full((self.N,), fill_value=np.inf) # (N,)
        best_count = np.zeros((self.N,), dtype=int) # (N,)
        return pos, vel, best_pos, best_score, best_count
    
    def _initPosRandom(
        self,
    ):
        """
        Initialize particles randomly.
        Returns:
            pos: random particle position; np.array (N, M)
        """
        return self.rng.random(size=(self.N, self.M))
    
    def _initVelRandom(
        self,
    ):
        """
        Initialize particles randomly.
        Returns:
            vel: random particle velocity; np.array (N, M)
        """
        vel = 2 * (self.rng.random(size=(self.N, self.M)) - 0.5)
        return 0.5 * np.sqrt(self.M) * vel
    
    def _updateParticle(
        self,
        n:int,
    ):
        """
        Update particle.
        Args:
            n: index of particle; int
        """
        # determine best neighbour
        dists = np.sum((self.pos - self.pos[n])**2, axis=-1) # (N,)
        neighbours = np.argsort(dists)[:self.num_neighbours+1] # (num_neighbours,)
        best_neighbour = np.argmin(self.best_score[neighbours])
        best_pos_neighbourhood = self.best_pos[neighbours[best_neighbour]]

        # calculate velocity and position of current particle
        vel = self.alpha_momentum * self.vel[n] \
            + self.alpha_propre * self.rng.random() * (self.best_pos[n] - self.pos[n]) \
            + self.alpha_social * self.rng.random() * (best_pos_neighbourhood - self.pos[n])
        pos = self.pos[n] + vel

        # limit position to [0, 1] and reflect velocity if position is out of bounds
        vel = np.where(((pos < 0) | (pos > 1)), -vel, vel)
        pos = np.clip(pos, 0, 1)
        
        # update velocity and position of current particle
        self.vel[n] = vel
        self.pos[n] = pos
        
    def _increaseIterator(
        self,
    ):
        """
        Increase particle iterator.
        """
        if self.n == self.N - 1:
            self.n = 0
        else:
            self.n += 1

    def _decreaseIterator(
        self,
    ):
        """
        Decrease particle iterator.
        """
        if self.n == 0:
            self.n = self.N - 1
        else:
            self.n -= 1
    
