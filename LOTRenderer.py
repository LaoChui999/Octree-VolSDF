import torch
import numpy as np
from torch import nn, autograd
from collections import namedtuple
from warnings import warn

from svox.helpers import _get_c_extension, LocalIndex, DataFormat
from svox.renderer import VolumeRenderer
from tqdm import tqdm

NDCConfig = namedtuple('NDCConfig', ["width", "height", "focal"])
Rays = namedtuple('Rays', ["origins", "dirs", "viewdirs"])

_C = _get_c_extension()


def _rays_spec_from_rays(rays):
    spec = _C.RaysSpec()
    spec.origins = rays.origins
    spec.dirs = rays.dirs
    spec.vdirs = rays.viewdirs
    return spec


class _VolumeRenderFunctionLOT(autograd.Function):
    @staticmethod
    def forward(ctx, data_density, data_sh, tree, rays, opt):
        out = _C.volume_render_LOT(tree, rays, opt)
        ctx.tree = tree
        ctx.rays = rays
        ctx.opt = opt
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_input_grad[0]:
            grad_holder = _C.OutputGradsLOT()

            _C.volume_render_backward_LOT(
                ctx.tree, ctx.rays, ctx.opt, grad_out.contiguous(), grad_holder)

            grad_density = grad_holder.grad_density_out
            grad_sh = grad_holder.grad_sh_out

            return grad_density, grad_sh, None, None, None


class LOTRenderA(VolumeRenderer):
    def __init__(self, tree,
                 step_size: float = 1e-3,
                 background_brightness: float = 1.0,
                 ndc: NDCConfig = None,
                 min_comp: int = 0,
                 max_comp: int = -1,
                 density_softplus: bool = False,
                 rgb_padding: float = 0.0, ):

        super(LOTRenderA, self).__init__(tree, step_size, background_brightness, ndc, min_comp,
                                         max_comp,
                                         density_softplus,
                                         rgb_padding)

    def RenderLaunch(self, rays: Rays):
        def dda_unit(cen, invdir):
            """
            voxel aabb ray tracing step
            :param cen: jnp.ndarray [B, 3] center
            :param invdir: jnp.ndarray [B, 3] 1/dir
            :return: tmin jnp.ndarray [B] at least 0;
                     tmax jnp.ndarray [B]
            """
            B = invdir.shape[0]
            tmin = torch.zeros((B,), dtype=cen.dtype, device=cen.device)
            tmax = torch.full((B,), fill_value=1e9, dtype=cen.dtype, device=cen.device)
            for i in range(3):
                t1 = -cen[..., i] * invdir[..., i]
                t2 = t1 + invdir[..., i]
                tmin = torch.max(tmin, torch.min(t1, t2))
                tmax = torch.min(tmax, torch.max(t1, t2))
            return tmin, tmax

        origins, dirs, viewdirs = rays.origins, rays.dirs, rays.viewdirs
        origins = self.tree.world2tree(origins)

        dirs = self.tree.invradius[None] * dirs
        delta_scale = 1.0 / (dirs.norm(dim=1))
        dirs = delta_scale[None].T * dirs

        B = dirs.size(0)
        assert viewdirs.size(0) == B and origins.size(0) == B
        # dirs /= torch.norm(dirs, dim=-1, keepdim=True)

        sh_mult = None
        if self.data_format.format == DataFormat.SH:
            from svox import sh
            sh_order = int(self.data_format.basis_dim ** 0.5) - 1
            sh_mult = sh.eval_sh_bases(sh_order, viewdirs)[:, None]

        invdirs = 1.0 / (dirs + 1e-9)
        t, tmax = dda_unit(origins, invdirs)
        light_intensity = torch.ones(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)

        good_indices = torch.arange(B, device=origins.device)
        # delta_scale = (dirs / self.tree.invradius[None]).norm(dim=1)
        a = 0
        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs

            treeview = self.tree[LocalIndex(pos)]
            leaf_node = torch.stack(treeview.key, dim=-1)

            node_corners = self.tree.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
            leaf_node[:, 2], leaf_node[:, 3]]
            node_num = node_corners.shape[0]
            node_corners = node_corners.view([node_num * 8])
            nc_d = self.tree.CornerD[node_corners.long()]
            nc_d = nc_d.view([node_num, 8])
            nc_sh = self.tree.CornerSH[node_corners.long()]
            nc_sh = nc_sh.view([node_num, 8, -1])

            cube_sz = treeview.lengths_local
            low_pos = treeview.corners_local

            rgb_t, sigma_t, pos_t = self.tree.TrilinearInterpolation(nc_d, nc_sh, pos, low_pos, cube_sz)
            temp_rgb = rgb_t.detach().cpu().numpy()
            # rgb_t= torch.zeros((origins.shape[0], 27), device=origins.device)
            # sigma_t= torch.ones((origins.shape[0], 1), device=origins.device)

            """
            chunk_size = 80000
            all_sigma = []
            all_rgb = []
            for i in tqdm(range(0, pos.shape[0], chunk_size)):
                treeview = self.tree[LocalIndex(pos[i:i + chunk_size])]
                leaf_node = torch.stack(treeview.key, dim=-1)

                node_corners = self.tree.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
                leaf_node[:, 2], leaf_node[:, 3]]
                node_num = node_corners.shape[0]
                node_corners = node_corners.view([node_num * 8, 1])
                nc_data = self.tree.CornerData[node_corners.long()]
                nc_data = nc_data.view([node_num, 8, -1])
                rgb_t, sigma_t = self.tree.TrilinearInterpolation(nc_data, pos[i:i + chunk_size], treeview)

                all_sigma.append(sigma_t)
                all_rgb.append(rgb_t)
            

            sigma_t = torch.cat(all_sigma, dim=0)
            rgb_t = torch.cat(all_rgb, dim=0)
            del all_sigma, all_rgb
            """

            # rgba = treeview.values
            # cube_sz = treeview.lengths_local
            # pos_t = (pos - low_pos) / cube_sz[:, None]
            treeview = None

            subcube_tmin, subcube_tmax = dda_unit(pos_t, invdirs)

            delta_t = (((subcube_tmax - subcube_tmin) * cube_sz)) + self.step_size
            # att = torch.exp(- delta_t * torch.relu(rgba[..., -1]) * delta_scale[good_indices])
            att = torch.exp(- delta_t * torch.relu(sigma_t[..., -1]) * delta_scale[good_indices])
            weight = light_intensity[good_indices] * (1.0 - att)
            # rgb = rgba[:, :-1]
            rgb = rgb_t
            if self.data_format.format != DataFormat.RGBA:
                # [B', 3, n_sh_coeffs]
                rgb_sh = rgb.reshape(-1, 3, self.data_format.basis_dim)
                rgb = torch.sigmoid(torch.sum(sh_mult * rgb_sh, dim=-1))  # [B', 3]
            else:
                rgb = torch.sigmoid(rgb)
            rgb = weight[:, None] * rgb[:, :3]

            out_rgb[good_indices] += rgb
            light_intensity[good_indices] *= att
            t += delta_t

            mask = t < tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            invdirs = invdirs[mask]
            t = t[mask]
            if sh_mult is not None:
                sh_mult = sh_mult[mask]
            tmax = tmax[mask]
            a += 1
        light_intensity = light_intensity.repeat(3, 1).T
        out_rgb += light_intensity * self.background_brightness
        return out_rgb

    def RenderLaunchCuda(self, rays, fast=False):
        return _VolumeRenderFunctionLOT.apply(
            self.tree.CornerD,
            self.tree.CornerSH,
            self.tree._LOTspec(),
            _rays_spec_from_rays(rays),
            self._get_options(fast))

    @staticmethod
    def RandomRays(c2w, width=800, height=800, fx=1111.111, fy=None, select_index=None):
        if fy is None:
            fy = fx
        origins = c2w[None, :3, 3].expand(height * width, -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=c2w.device),
            torch.arange(width, dtype=torch.float64, device=c2w.device),
        )
        xx = (xx - width * 0.5) / float(fx)
        yy = (yy - height * 0.5) / float(fy)
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, -yy, -zz), dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3)
        del xx, yy, zz
        dirs = torch.matmul(c2w[None, :3, :3].double(), dirs[..., None])[..., 0].float()

        if (select_index is not None):
            origins = torch.index_select(origins, dim=0, index=select_index)
            dirs = torch.index_select(dirs, dim=0, index=select_index)

        vdirs = dirs

        return Rays(
            origins=origins,
            dirs=dirs,
            viewdirs=vdirs
        )

    @staticmethod
    def GenerateRaysByCamera(c2w, fx, fy, cx, cy, width, height):
        # origins = c2w[None, :3, 3].expand(int(height * width), -1).contiguous()
        #
        # yy, xx = torch.meshgrid(
        #     torch.arange(height, dtype=torch.float32),
        #     torch.arange(width, dtype=torch.float32),
        # )
        #
        # xx = (xx - cx) / fx
        # yy = (yy - cy) / fy
        #
        # zz = torch.ones_like(xx)
        # dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        # # dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        # dirs = dirs.reshape(1, -1, 3, 1)
        # del xx, yy, zz
        #
        # dirs=dirs.cuda()
        # dirs = (c2w[None, :3, :3] @ dirs)[..., 0]
        # dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        #
        # dirs = dirs.reshape(-1, 3).float()
        #
        # vdirs = dirs
        #
        # return Rays(
        #     origins=origins,
        #     dirs=dirs,
        #     viewdirs=vdirs
        # )


        origins = c2w[None, :3, 3].expand(int(height * width), -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=c2w.device) + 0.5,
            torch.arange(width, dtype=torch.float64, device=c2w.device) + 0.5,
        )
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()

        vdirs = dirs

        return Rays(
            origins=origins,
            dirs=dirs,
            viewdirs=vdirs
        )

    def SelectRender(self, c2w, width=800, height=800, fx=1111.111, fy=None, select_index=None):
        return self.RenderLaunch(LOTRenderA.RandomRays(c2w, width, height, fx, fy, select_index))

    def RenderImage(self, c2w, fx, fy, cx, cy, width, height):
        return self.RenderLaunch(LOTRenderA.GenerateRaysByCamera(c2w, fx, fy, cx, cy, width, height))

    def SelectRenderCuda(self, c2w, width=800, height=800, fx=1111.111, fy=None, select_index=None):
        return self.RenderLaunchCuda(LOTRenderA.RandomRays(c2w, width, height, fx, fy, select_index))

    def SelectRenderRaysCuda(self, ray_origins, ray_dirs):
        return self.RenderLaunchCuda(Rays(origins=ray_origins, dirs=ray_dirs, viewdirs=ray_dirs))
