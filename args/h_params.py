

class HParams():
    def __init__(
        self, 
        name,
    ):
        self.self_name = name

    def setHParams(
        self, 
        hparams:dict,
    ):
        """
        Set hyper parameters from a dictionary
        Args:
            hparams: dictionary with hyper parameters; dict
        """
        for key in self.__dict__.keys():
            if key != "self_name":
                setattr(self, key, hparams[self.self_name][key])

    def getHParams(
        self,
    ):
        """
        Get hyper parameters as dictionary
        Returns:
            dictionary with hyper parameters; dict
        """
        self_dict = {}
        for key, value in self.__dict__.items():
            if key != "self_name":
                self_dict[key] = value
        return self_dict


class HParamsDataset(HParams):
    def __init__(self) -> None:
        # hyper parameters
        self.name = None
        self.split_ratio = None
        self.keep_N_observations = None
        self.keep_sensor = None
        self.sensors = None

        HParams.__init__(self, name="dataset")


class HParamsModel(HParams):
    def __init__(self) -> None:
        # hyper parameters
        self.ckpt_path = None
        self.scale = None
        self.encoder_type = None
        self.hash_levels = None
        self.hash_max_res = None
        self.grid_type = None
        self.save = None

        HParams.__init__(self, name="model")
     

class HParamsTraining(HParams):
    def __init__(self) -> None:
        # hyper parameters
        self.batch_size = None
        self.sampling_strategy = None
        self.sensors = None
        self.max_steps = None
        self.max_time = None
        self.lr = None
        self.rgbd_loss_w = None
        self.tof_loss_w = None
        self.uss_loss_w = None
        self.color_loss_w = None
        self.debug_mode = None
        self.real_time_simulation = None

        HParams.__init__(self, name="training")
    
    def checkArgs(self):
        if self.sampling_strategy["imgs"] == "all" and self.sampling_strategy["pixs"] != "random":
            self.sampling_strategy["pixs"] = "random"
            print(f"WARNING: HParamsTraining:checkArgs: sampling strategy for rays must be 'random' if sampling strategy for images is 'all' ")


class HParamsEvaluation(HParams):
    def __init__(self) -> None:
        # hyper parameters
        self.batch_size = None
        self.res_map = None
        self.res_angular = None
        self.eval_every_n_steps = None
        self.num_color_pts = None
        self.num_depth_pts = None
        self.num_depth_pts_per_step = None
        self.num_plot_pts = None
        self.height_tolerance = None
        self.density_map_thr = None
        self.inlier_threshold = None
        self.zones = None
        self.sensors = None
        self.plot_results = None

        HParams.__init__(self, name="evaluation")


class HParamsNGPGrid(HParams):
    def __init__(self) -> None:
        # hyper parameters
        self.update_interval = None
        self.warmup_steps = None

        HParams.__init__(self, name="ngp_grid")


class HParamsOccGrid(HParams):
    def __init__(self) -> None:
        # hyper parameters
        self.batch_size = None
        self.update_interval = None
        self.decay_warmup_steps = None
        self.batch_ratio_ray_update = None
        self.false_detection_prob_every_m = None
        self.std_every_m = None
        self.nerf_pos_noise_every_m = None
        self.nerf_threshold_max = None
        self.nerf_threshold_slope = None

        HParams.__init__(self, name="occ_grid")


class HParamsETHZ(HParams):
    def __init__(self) -> None:
        # hyper parameters
        self.dataset_dir = None
        self.room = None
        self.cam_ids = None
        self.use_optimized_poses = None

        HParams.__init__(self, name="ethz")


class HParamsRobotAtHome(HParams):
    def __init__(self) -> None:
        # hyper parameters
        self.dataset_dir = None
        self.session = None
        self.home = None
        self.room = None
        self.subsession = None
        self.home_session = None

        HParams.__init__(self, name="RH2")


class HParamsRGBD(HParams):
    def __init__(self):
        # hyper parameters
        self.angle_of_view = None

        HParams.__init__(self, name="RGBD")


class HParamsUSS(HParams):
    def __init__(self):
        # hyper parameters
        self.angle_of_view = None

        HParams.__init__(self, name="USS")


class HParamsToF(HParams):
    def __init__(self):
        # hyper parameters
        self.angle_of_view = None
        self.matrix = None
        self.tof_pix_size = None
        self.sensor_calibration_error = None
        self.sensor_random_error = None

        HParams.__init__(self, name="ToF")


class HParamsLiDAR(HParams):
    def __init__(self):
        # hyper parameters
        self.angle_min_max = None

        HParams.__init__(self, name="LiDAR")