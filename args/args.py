import torch
import numpy as np
import random
import os
import json
from datetime import datetime
import shutil
import logging

from args.h_params import HParamsDataset, HParamsModel, HParamsTraining, \
                            HParamsEvaluation, HParamsNGPGrid, HParamsOccGrid, HParamsRobotAtHome, \
                            HParamsRGBD, HParamsUSS, HParamsToF, HParamsETHZ, HParamsLiDAR
from args.logging_formatter import FileFormatter, TerminalFormatter


class Args():
    def __init__(
        self, 
        file_name
    ) -> None:
        
        # hyper parameters
        self.dataset = HParamsDataset()
        self.model = HParamsModel()
        self.training = HParamsTraining()
        self.eval = HParamsEvaluation()
        self.occ_grid = HParamsOccGrid()

        # set hyper parameters
        hparams = self.readJson(file_name)
        self.dataset.setHParams(hparams)
        self.model.setHParams(hparams)
        self.training.setHParams(hparams)
        self.eval.setHParams(hparams)
        self.occ_grid.setHParams(hparams)

        if self.dataset.name == "ETHZ":
            self.ethz = HParamsETHZ()
            self.ethz.setHParams(hparams)

            if self.model.grid_type == "ngp":
                self.ngp_grid = HParamsNGPGrid()
                self.ngp_grid.setHParams(hparams)
            elif self.model.grid_type == "occ":
                self.occ_grid = HParamsOccGrid()
                self.occ_grid.setHParams(hparams)
            else:
                self.logger.error("Grid type not implemented!")

        elif self.dataset.name == "RH2":
            self.rh = HParamsRobotAtHome()
            self.rh.setHParams(hparams)

            self.ngp_grid = HParamsNGPGrid()
            self.ngp_grid.setHParams(hparams)
        else:
            self.logger.error("Dataset not implemented!")

        self.rgbd = HParamsRGBD()
        self.rgbd.setHParams(hparams)
        self.uss = HParamsUSS()
        self.uss.setHParams(hparams)
        self.tof = HParamsToF()
        self.tof.setHParams(hparams)    
        self.lidar = HParamsLiDAR()
        self.lidar.setHParams(hparams)        

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # random seed
        self.seed = 21
        self.setRandomSeed(
            seed=self.seed,
        )

        # create saving directory
        self.createSaveDir()

        # initialize logging
        self._initLogging()

        # rendering configuration
        self.exp_step_factor = 1 / 256 if self.model.scale > 0.5 else 0. 

    def setRandomSeed(
        self,
        seed:int,
    ):
        """
        Set random seed
        Args:
            seed: random seed; int
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True

    def createSaveDir(
        self,
    ):
        """
        Create saving directory
        """
        t = datetime.now()
        time_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M%S")
        self.save_dir = os.path.join('results/', self.dataset.name, time_name)
        if not os.path.exists('results/'):
            os.mkdir('results/')
        if not os.path.exists(os.path.join('results/', self.dataset.name)):
            os.mkdir(os.path.join('results/', self.dataset.name))
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.mkdir(self.save_dir)

    def readJson(
        self, 
        file_name,
    ):
        """
        Read hyper parameters from json file
        Args:
            file_name: name of json file; str
        Returns:
            hparams: hyper parameters; dict
        """
        file_path = os.path.join("args", file_name)
        with open(file_path) as f:
            hparams = json.load(f)

        return hparams
    
    def saveJson(
        self,
    ):
        """
        Save arguments in json file
        """
        hparams = {}
        hparams["dataset"] = self.dataset.getHParams()
        hparams["model"] = self.model.getHParams()
        hparams["training"] = self.training.getHParams()
        hparams["occ_grid"] = self.occ_grid.getHParams()
        hparams["RGBD"] = self.rgbd.getHParams()
        hparams["USS"] = self.uss.getHParams()
        hparams["ToF"] = self.tof.getHParams()
        hparams["LiDAR"] = self.lidar.getHParams()

        if self.dataset.name == "RH2":
            hparams["RH2"] = self.rh.getHParams()
        elif self.dataset.name == "ETHZ":
            hparams["ETHZ"] = self.ethz.getHParams()
                    
        
        # Serializing json
        json_object = json.dumps(hparams, indent=4)
        
        # Writing to sample.json
        with open(os.path.join(self.save_dir, "hparams.json"), "w") as outfile:
            outfile.write(json_object)

    def _initLogging(
        self,
    ):
        """
        Create logger
        Returns:
            logger: logger; logging.Logger
        """
        # Create a custom logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(self.save_dir, "log.txt"))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = TerminalFormatter()
        f_format = FileFormatter()
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)






