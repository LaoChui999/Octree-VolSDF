import torch
import math
from torch import nn, autograd
from typing import Union, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from svox2 import utils

_C = utils._get_c_extension()


@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, renderer backend
    :param background_brightness: float
    :param step_size: float, step size for rendering
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    backend: str = "cuvol"

    background_brightness: float = 1.0

    step_size: float = 0.0

    t_step: float = 2.0

    sigma_thresh: float = 1e-8  # Voxels with sigmas < this are ignored, in [0, 1]
    #  make this higher for fast rendering

    sdf_thresh: float = 0.7

    alpha_thresh: float = 1e-10

    stop_thresh: float = (
        1e-10  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    last_sample_opaque: bool = False  # Make the last sample opaque (for forward-facing)

    near_clip: float = 0.0
    use_spheric_clip: bool = False

    random_sigma_std: float = 1.0  # Noise to add to sigma (only if randomize=True)
    random_sigma_std_background: float = 1.0  # Noise to add to sigma

    # (for the BG model; only if randomize=True)

    step_sigma: float = 0.04
    step_max: float = 2.0
    step_min: float = 2.0
    step_K: float = 1.0
    step_b: float = 0.0

    sample_size: int = 64
    cube_thresh: float = 64

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_size = self.step_size
        opt.t_step = self.t_step
        opt.sigma_thresh = self.sigma_thresh
        opt.sdf_thresh = self.sdf_thresh
        opt.alpha_thresh = self.alpha_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip

        opt.last_sample_opaque = self.last_sample_opaque

        opt.step_sigma = self.step_sigma
        opt.step_K = self.step_K
        opt.step_b = self.step_b

        opt.sample_size = self.sample_size
        opt.cube_thresh = self.cube_thresh

        #  opt.randomize = randomize
        #  opt.random_sigma_std = self.random_sigma_std
        #  opt.random_sigma_std_background = self.random_sigma_std_background

        #  if randomize:
        #      # For our RNG
        #      UINT32_MAX = 2**32-1
        #      opt._m1 = np.random.randint(0, UINT32_MAX)
        #      opt._m2 = np.random.randint(0, UINT32_MAX)
        #      opt._m3 = np.random.randint(0, UINT32_MAX)
        #      if opt._m2 == opt._m3:
        #          opt._m3 += 1  # Prevent all equal case
        # Note that the backend option is handled in Python
        return opt

    def ComGaussianStep(self):
        x_1 = (1.0 / (self.step_sigma * pow(2.0 * 3.1415926, 0.5))) * math.exp(
            -(1.0 * 1.0) / (2.0 * self.step_sigma * self.step_sigma))
        x_2 = 1.0 / (self.step_sigma * pow(2.0 * 3.1415926, 0.5))

        self.step_K = (self.step_max - self.step_min) / (x_2 - x_1)
        self.step_b = self.step_max - self.step_K * x_2

        f = self.step_K * (1.0 / (self.step_sigma * pow(2.0 * 3.1415926, 0.5))) * math.exp(
            -(0.2 * 0.2) / (2.0 * self.step_sigma * self.step_sigma)) + self.step_b
        a = self.step_K * (1.0 / (self.step_sigma * pow(2.0 * 3.1415926, 0.5))) * math.exp(
            -(0.1 * 0.1) / (2.0 * self.step_sigma * self.step_sigma)) + self.step_b
        b = self.step_K * (1.0 / (self.step_sigma * pow(2.0 * 3.1415926, 0.5))) * math.exp(
            -(0.05 * 0.05) / (2.0 * self.step_sigma * self.step_sigma)) + self.step_b
        e = self.step_K * (1.0 / (self.step_sigma * pow(2.0 * 3.1415926, 0.5))) * math.exp(
            -(0.02 * 0.02) / (2.0 * self.step_sigma * self.step_sigma)) + self.step_b
        c = self.step_K * (1.0 / (self.step_sigma * pow(2.0 * 3.1415926, 0.5))) * math.exp(
            -(0.01 * 0.01) / (2.0 * self.step_sigma * self.step_sigma)) + self.step_b
        d = self.step_K * (1.0 / (self.step_sigma * pow(2.0 * 3.1415926, 0.5))) * math.exp(
            -(0.0 * 0.0) / (2.0 * self.step_sigma * self.step_sigma)) + self.step_b

        a = self.step_K


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def is_cuda(self) -> bool:
        return self.origins.is_cuda and self.dirs.is_cuda


@dataclass
class RaysHitLOTSDF:
    sdf_point: torch.Tensor
    col_point: torch.Tensor
    hitnode_sdf: torch.Tensor
    hitnode_col: torch.Tensor
    hitnum: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.RaysHitLOTreeSDF()
        spec.sdf_point = self.sdf_point
        spec.col_point = self.col_point
        spec.hitnode_sdf = self.hitnode_sdf
        spec.hitnode_col = self.hitnode_col
        spec.hitnum = self.hitnum
        return spec

    def __getitem__(self, key):
        return RaysHitLOTSDF(self.sdf_point[key], self.col_point[key],
                             self.hitnode_sdf[key], self.hitnode_col[key],
                             self.hitnum[key])


@dataclass
class Camera:
    c2w: torch.Tensor  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, -1.0)

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w
        spec.fx = self.fx_val
        spec.fy = self.fy_val
        spec.cx = self.cx_val
        spec.cy = self.cy_val
        spec.width = self.width
        spec.height = self.height
        spec.ndc_coeffx = self.ndc_coeffs[0]
        spec.ndc_coeffy = self.ndc_coeffs[1]
        return spec

    @property
    def is_cuda(self) -> bool:
        return self.c2w.is_cuda

    def gen_rays(self) -> Rays:
        """
        Generate the rays for this camera
        :return: (origins (H*W, 3), dirs (H*W, 3))
        """
        origins = self.c2w[None, :3, 3].expand(int(self.height) * int(self.width), -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float64, device=self.c2w.device) + 0.5,
            torch.arange(self.width, dtype=torch.float64, device=self.c2w.device) + 0.5,
        )
        xx = (xx - self.cx_val) / self.fx_val
        yy = (yy - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (self.c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()

        if self.ndc_coeffs[0] > 0.0:
            origins, dirs = utils.convert_to_ndc(
                origins,
                dirs,
                self.ndc_coeffs)
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        return Rays(origins, dirs)
