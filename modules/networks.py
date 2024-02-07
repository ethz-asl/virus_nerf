from typing import Callable, Optional
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd


from .triplane import TriPlaneEncoder
from .volume_train import VolumeRenderer
from .spherical_harmonics import DirEncoder
from modules.occupancy_grid import OccupancyGrid
from modules.ngp_grid import NGPGrid
from datasets.scene_base import SceneBase
from datasets.dataset_rh import DatasetRH
from args.args import Args

class TruncExp(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))


class NGP(nn.Module):

    def __init__(
            self, 
            scale: float=0.5, 
            # position encoder config
            pos_encoder_type: str='hash', 
            levels: int=16, # number of levels in hash table
            feature_per_level: int=2, # number of features per level
            log2_T: int=19, # maximum number of entries per level 2^19
            base_res: int=16, # minimum resolution of  hash table
            max_res: int=2048, # maximum resolution of the hash table
            half_opt: bool=False, # whether to use half precision, available for hash
            # mlp config
            xyz_net_width: int=64,
            xyz_net_depth: int=1,
            xyz_net_out_dim: int=16,
            rgb_net_depth: int=2,
            rgb_net_width: int=64,
            scene:SceneBase=None,
            dataset:DatasetRH=None,
            args:Args=None,
        ):
        super().__init__()

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128

        if pos_encoder_type == 'hash':
            if half_opt:
                from .hash_encoder_half import HashEncoder
            else:
                from .hash_encoder import HashEncoder

            self.pos_encoder = HashEncoder(
                max_params=2**log2_T,
                base_res=base_res,
                max_res=max_res,
                levels=levels,
                feature_per_level=feature_per_level,
            )
        elif pos_encoder_type == 'triplane':
            self.pos_encoder = TriPlaneEncoder(
                base_res=16,
                max_res=max_res,
                levels=8,
                feature_per_level=4,
            )
        else:
            self.args.logger.error(f"pos_encoder_type {pos_encoder_type} not implemented")

        self.xyz_encoder = MLP(
            input_dim=self.pos_encoder.out_dim,
            output_dim=xyz_net_out_dim,
            net_depth=xyz_net_depth,
            net_width=xyz_net_width,
            bias_enabled=False,
        )

        self.dir_encoder = DirEncoder()

        rgb_input_dim = (
            self.dir_encoder.out_dim + \
            self.xyz_encoder.output_dim
        )
        self.rgb_net =  MLP(
            input_dim=rgb_input_dim,
            output_dim=3,
            net_depth=rgb_net_depth,
            net_width=rgb_net_width,
            bias_enabled=False,
            output_activation=nn.Sigmoid()
        )

        self.render_func = VolumeRenderer()

        self.args = args
        if self.args.model.grid_type == 'ngp':
            self.occupancy_grid = NGPGrid(
                args=args,
                grid_size=self.grid_size,
                fct_density=self.density,
            )
        elif self.args.model.grid_type == 'occ':
            self.occupancy_grid = OccupancyGrid(
                args=args,
                grid_size=self.grid_size,
                scene=scene,
                dataset=dataset,
                fct_density=self.density,
            )
        else:
            self.args.logger.error(f"grid_type {self.args.model.grid_type} not implemented")

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature
        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        embedding = self.pos_encoder(x)
        h = self.xyz_encoder(embedding)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat:
            return sigmas, h
        return sigmas

    def forward(self, x, d):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions
        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        return sigmas, rgbs

    @torch.no_grad()
    def updateNeRFGrid(
        self,
        density_threshold,
        warmup=False,
        decay=0.95,
        erode=False       
    ):

        self.occupancy_grid.update(
            density_threshold=density_threshold,
            warmup=warmup,
            decay=decay,
            erode=erode,       
        )

    @torch.no_grad()
    def updateOccGrid(
        self,
        density_threshold:float,
        elapse_time:float,   
    ):

        self.occupancy_grid.update(
            elapse_time=elapse_time,
        )



class MLP(nn.Module):
    '''
        A simple MLP with skip connections from:
        https://github.com/KAIR-BAIR/nerfacc/blob/master/examples/radiance_fields/mlp.py
    '''

    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled))
            if ((self.skip_layer is not None) and (i % self.skip_layer == 0)
                    and (i > 0)):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(in_features,
                                          self.output_dim,
                                          bias=bias_enabled)
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):

        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    # @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if ((self.skip_layer is not None) and (i % self.skip_layer == 0)
                    and (i > 0)):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x
