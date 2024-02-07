import numpy as np
import pandas as pd
import os


class Metric():
    def __init__(
        self,
        metric_name:str,
        hparams_lims:np.ndarray,
        rng:np.random.Generator,
        save_dir:str,
    ) -> None:
        """
        Args:
            metric_name: name of metric; str
            hparams_lims: limits of hparams; np.array (M, 2)
            rng: random number generator; np.random.Generator
            save_dir: directory to save parameters; str
        """
        self.name = metric_name
        self.rng = rng

        # load parameters if they exist otherwise set them
        metric_path = os.path.join(save_dir, "metrics_params.csv")
        if os.path.exists(metric_path):
            self.centre, self.std, self.freq, self.rand_std = self._loadParams(
                metric_path=metric_path
            )
        else:
            self.centre, self.std, self.freq, self.rand_std = self._setParams(
                hparams_lims=hparams_lims,
            )
            self._saveParams(
                metric_path=metric_path
            )

    def __call__(
        self,
        X:np.ndarray,
    ):
        """
        Evaluate metric.
        Args:
            X: position in hparams space; np.array (M,) or (N, M)
        Returns:
            score: score of input; float or np.array (N,)
        """
        if self.name == "gauss":
            return self.gauss(
                X=X,
            ) # float  
        elif self.name == "cos":
            return self.cos(
                X=X,
            ) # float
        elif self.name == "rand":
            return self.rand(
                X=X,
            ) # float
        else:
            print(f"ERROR: Metric.__call__: metric_name {self.name} not supported")

    def gauss(
        self,
        X:np.ndarray,
    ):
        """
        Evaluate gaussian.
        Args:
            X: position in hparams space; np.array (M,) or (N, M)
        Returns:
            score: score of input; float or np.array (N,)
        """
        score_inv = np.exp(- np.sum((X-self.centre)**2 / self.std**2, axis=-1))
        score = 1 - score_inv
        return score

    def cos(
        self,
        X:np.ndarray,
    ):
        """
        Evaluate cosine-gaussian.
        Args:
            X: position in hparams space; np.array (M,) or (N, M)
        Returns:
            score: score of input; float or np.array (N,)
        """
        exp_score = self.gauss(
            X=X,
        ) # float
        cos_score_inv = np.prod((np.cos(2*np.pi * self.freq * (X-self.centre))+1)/2, axis=-1)
        cos_score = 1 - cos_score_inv
        return cos_score * exp_score
    
    def rand(
        self,
        X:np.ndarray,
    ):
        """
        Evaluate random metric.
        Args:
            X: position in hparams space; np.array (M,) or (N, M)
        Returns:
            score: score of input; float or np.array (N,)
        """
        score = self.cos(
            X=X,
        ) # float
        rand_score = self.rng.normal(0, self.rand_std)
        return np.clip(score + rand_score, 0, 1)
    
    def _setParams(
        self,
        hparams_lims:np.ndarray,
    ):
        """
        Set metric parameters.
        Args:
            hparams_lims: limits of hparams; np.array (M, 2)
        Returns:
            centre: centre of gaussian; np.array (M,)
            std: standard deviation of gaussian; float
            freq: frequency of cosine; float
            rand_std: standard deviation of random noise; float
        """
        delta = hparams_lims[:, 1]-hparams_lims[:, 0]
        centre = self.rng.uniform(hparams_lims[:, 0], hparams_lims[:, 1])
        std = self.rng.uniform(delta/6, delta/3)
        freq = self.rng.uniform(delta/6, delta)
        rand_std = 0.1 #self.rng.uniform(0.1, 0.8)
        return centre, std, freq, rand_std

    def _saveParams(
        self,
        metric_path:str,
    ):
        """
        Save metric parameters.
        Args:
            metric_path: path to save parameters; str
        """
        params = {
            "name": self.centre.shape[0] * [self.name],
            "centre": self.centre,
            "std": self.std,
            "freq": self.freq,
            "rand_std": self.centre.shape[0] * [self.rand_std],
        }
        pd.DataFrame(
            data=params,
        ).to_csv(metric_path, index=False)

    def _loadParams(
        self,
        metric_path:str,
    ):
        """
        Load metric parameters.
        Args:
            metric_name: name of metric; str
            metric_path: path to save parameters; str
        Returns:
            centre: centre of gaussian; np.array (M,)
            std: standard deviation of gaussian; float
            freq: frequency of cosine; float
            rand_std: standard deviation of random noise; float
        """
        params = pd.read_csv(metric_path)
        metric_name = params["name"][0]
        centre = params["centre"].to_numpy()
        std = params["std"].to_numpy()
        freq = params["freq"].to_numpy()
        rand_std = params["rand_std"][0]

        if metric_name != self.name:
            print(f"ERROR: Metric.loadParams: metric_name {metric_name} does not match {self.name}")

        return centre, std, freq, rand_std