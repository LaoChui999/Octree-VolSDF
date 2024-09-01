import gc

import torch
import numpy as np
import math
import mcubes
import trimesh
import os

from torch import nn, autograd, cuda
from svox.helpers import N3TreeView, DataFormat, LocalIndex, _get_c_extension

from svox.svox import N3Tree
from svox2 import utils, defs
from LONode import LONodeA
from LOTCorr import RenderOptions, Rays, RaysHitLOTSDF

from tqdm import tqdm
from typing import List, Optional
from collections import OrderedDict

from matplotlib import cm
import matplotlib.pyplot as plt
import mitsuba as mi
import imageio

# _C = _get_c_extension()
_C = utils._get_c_extension()


class LOctreeA(N3Tree):
    def __init__(self, N=2, data_dim=None, basis_dim=9, basis_type: int = defs.BASIS_TYPE_SH,
                 depth_limit=8, init_reserve=1, init_refine=0, geom_resize_fact=1.0,
                 radius=0.5, center=[0.5, 0.5, 0.5],
                 data_format="RGBA",
                 extra_data=None,
                 device="cpu",
                 dtype=torch.float32,
                 map_location=None):
        super(LOctreeA, self).__init__(N, data_dim, depth_limit,
                                       init_reserve, init_refine, geom_resize_fact,
                                       radius, center,
                                       data_format,
                                       extra_data,
                                       device,
                                       dtype,
                                       map_location)

        # for i in range(7):
        #     init_reserve += (2 ** i) ** 3
        # init_reserve = int(2.4e6)

        self.register_parameter("CornerSH", nn.Parameter(
            torch.zeros(1, self.data_dim - 1, dtype=dtype, device=device, requires_grad=False)))
        self.register_parameter("CornerD", nn.Parameter(
            torch.zeros(1, 1, dtype=dtype, device=device, requires_grad=False)))
        self.register_parameter("CornerSDF", nn.Parameter(
            torch.zeros(1, 1, dtype=dtype, device=device, requires_grad=False)))
        self.register_parameter("CornerGaussSDF", nn.Parameter(
            torch.zeros(1, 1, dtype=dtype, device=device, requires_grad=False)))
        self.register_parameter("LearnS", nn.Parameter(
            torch.zeros(1, 1, dtype=dtype, device=device, requires_grad=False)))
        self.register_parameter("Beta", nn.Parameter(
            torch.zeros(1, 1, dtype=dtype, device=device, requires_grad=False)))
        self.register_buffer("CornerIndex", -1 * torch.ones(
            1, 1, 1, dtype=torch.int32, device=device))
        self.register_buffer("NodeCorners", torch.zeros(
            1, N, N, N, 8, dtype=torch.int32, device=device))
        ########################################################
        self.register_buffer("NodeNeighbors", -1 * torch.ones(
            1, 6, dtype=torch.int32, device=device))
        self.register_buffer("NodeAllNeighbors", -1 * torch.ones(
            1, 6, dtype=torch.int32, device=device))
        self.register_buffer("NodeAllNeighLen", -1.0 * torch.ones(
            1, 6, dtype=torch.float32, device=device))
        ########################################################
        self.register_buffer("NodeGaussNeighbors", -1 * torch.ones(
            1, 5, 5, 5, dtype=torch.int32, device=device))
        self.register_buffer("NodeGaussKernals", -1 * torch.ones(
            1, 5, 5, 5, dtype=torch.float32, device=device))
        self.register_buffer("NodeGaussGradNeighbors", -1 * torch.ones(
            1, 3, 3, 3, dtype=torch.int32, device=device))
        self.register_buffer("NodeGaussGradKernals", -1 * torch.ones(
            1, 3, 3, 3, dtype=torch.float32, device=device))
        ########################################################
        self.register_buffer("NodeGhoNeighbors", -1 * torch.ones(
            1, 3, 4, dtype=torch.int32, device=device))
        self.register_buffer("NodeGhoCoeff", -1 * torch.ones(
            1, 3, 4, dtype=torch.float32, device=device))
        ########################################################
        self.register_buffer("CornerMap", -1 * torch.ones(
            0, dtype=torch.int32, device=device))
        self.register_buffer("LeafNodeMap", -1 * torch.ones(
            0, 2, 2, 2, 1, dtype=torch.int32, device=device))
        self.register_buffer("ValidGeoCorner", -1 * torch.ones(
            0, dtype=torch.int32, device=device))
        self.register_buffer("ValidGeoCornerCoord", -1 * torch.ones(
            0, 3, dtype=torch.int16, device=device))

        self.CornerDict = {}

        self.register_buffer("_n_corners", torch.tensor(0, device=device))
        self.register_buffer("_c_depth", torch.tensor(0, device=device))

        self.basis_dim = basis_dim
        self.basis_type = basis_type

        self.density_rms: Optional[torch.Tensor] = None
        self.sh_rms: Optional[torch.Tensor] = None
        self.sdf_rms: Optional[torch.Tensor] = None
        self.beta_rms: Optional[torch.Tensor] = None
        self.data_sh_rms: Optional[torch.Tensor] = None
        self.data_sdf_rms: Optional[torch.Tensor] = None
        self.opt = RenderOptions()

    def GetMorton(self, pos):
        def _expand_bits(v):
            v &= 0x1fffff
            v = (v | v << 32) & 0x1f00000000ffff
            v = (v | v << 16) & 0x1f0000ff0000ff
            v = (v | v << 8) & 0x100f00f00f00f00f
            v = (v | v << 4) & 0x10c30c30c30c30c3
            v = (v | v << 2) & 0x1249249249249249
            return v

        xx = _expand_bits(pos[:, 0].type(torch.long))
        yy = _expand_bits(pos[:, 1].type(torch.long))
        zz = _expand_bits(pos[:, 2].type(torch.long))
        return (xx << 2) + (yy << 1) + zz

    def DecodeMorton(self, morton):
        def _unexpand_bits(m):
            v = m & 0x1249249249249249
            v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3
            v = (v ^ (v >> 4)) & 0x100f00f00f00f00f
            v = (v ^ (v >> 8)) & 0x1f0000ff0000ff
            v = (v ^ (v >> 16)) & 0x1f00000000ffff
            v = (v ^ (v >> 32)) & 0x1fffff

            v.type(torch.int16)
            return v

        z = _unexpand_bits(morton)
        y = _unexpand_bits(morton >> 1)
        x = _unexpand_bits(morton >> 2)

        coord = torch.stack((x, y, z), dim=1)

        return coord

    def RebuildCornerDict(self, new_corner_pos):
        new_corner_pos = new_corner_pos.view(-1, 3)

        # out_mask_index = self.FromPosGetCornerIndex(new_corner_pos)
        # out_mask_index = (out_mask_index == -1)

        new_corner_pos = torch.unique(new_corner_pos, dim=0)

        exist_corner_index = self.FromPosGetCornerIndex(new_corner_pos)
        not_exist_corner_mask = (exist_corner_index == -1)
        new_corner_pos = new_corner_pos[not_exist_corner_mask.cpu()]

        new_corner_morton = self.GetMorton(new_corner_pos)
        new_corner_dict = {k: i + len(self.CornerDict) for i, k in enumerate(new_corner_morton.cpu().numpy())}

        self.CornerDict = OrderedDict(list(self.CornerDict.items()) + list(new_corner_dict.items()))

        self._n_corners += new_corner_pos.size(0)

        prev_t = self.CornerSH.clone()
        self.CornerSH = nn.Parameter(
            torch.zeros(self._n_corners, 27, dtype=torch.float32,
                        device=prev_t.device, requires_grad=True))
        self.CornerSH[0:prev_t.size(0)] = prev_t[:]

        prev_t = self.CornerSDF.clone()
        self.CornerSDF = nn.Parameter(
            torch.zeros(self._n_corners, 1, dtype=torch.float32,
                        device=prev_t.device, requires_grad=True))
        self.CornerSDF[0:prev_t.size(0)] = prev_t[:]
        del prev_t

        self.CornerMap = -1 * torch.ones(self._n_corners, dtype=torch.int32, device=self.CornerSH.device)

        return None

    def FromPosGetCornerIndex(self, corner_pos):
        corner_morton = self.GetMorton(corner_pos)

        # vectorized_get = np.vectorize(lambda i: self.CornerDict.get(i, -1))
        # corner_index = vectorized_get(corner_morton.cpu().numpy())
        corner_index = [self.CornerDict.get(i, -1) for i in corner_morton.cpu().numpy()]

        trinkets = []
        trinkets.extend(corner_index)
        trinkets = torch.tensor(trinkets, device=self.CornerSH.device, dtype=torch.int)

        return trinkets

    def FastFromPosGetCornerIndex(self, corner_pos):
        corner_morton = self.GetMorton(corner_pos)

        # vectorized_get = np.vectorize(lambda i: self.CornerDict.get(i, -1))
        # corner_index = vectorized_get(corner_morton.cpu().numpy())
        corner_index = [self.CornerDict[i] for i in corner_morton.cpu().numpy()]

        trinkets = []
        trinkets.extend(corner_index)
        trinkets = torch.tensor(trinkets, device=self.CornerSH.device, dtype=torch.int)

        return trinkets

    def FromCornerIndexGetPos(self, corner_index, dict):
        corner_morton = [dict[i] for i in corner_index.cpu().numpy()]

        trinkets = []
        trinkets.extend(corner_morton)
        trinkets = torch.tensor(trinkets, device=self.CornerSH.device, dtype=torch.long)

        corner_pos = self.DecodeMorton(trinkets)

        return corner_pos

    def SetGeoCorner(self, corner):
        inverse_dict = {v: k for k, v in self.CornerDict.items()}
        geo_corner_crood = self.FromCornerIndexGetPos(corner, inverse_dict)

        def custom_sort(coord):
            x, y, z = coord
            return (x, y, z)

        geo_corner_crood = sorted(geo_corner_crood.tolist(), key=custom_sort)
        self.ValidGeoCornerCoord = torch.tensor(geo_corner_crood, dtype=torch.int16, device=self.CornerSH.device)

        corner_index = self.FastFromPosGetCornerIndex(self.ValidGeoCornerCoord)
        self.ValidGeoCorner = torch.tensor(corner_index, dtype=torch.int32, device=self.CornerSH.device)

        if self.CornerMap.numel() == 0:
            self.CornerMap = -1 * torch.ones(self.CornerSH.size(0), dtype=torch.int32, device=self.CornerSH.device)

        self.CornerMap[self.ValidGeoCorner.long()] = torch.arange(0, self.ValidGeoCorner.size(0), dtype=torch.int32,
                                                                  device=self.CornerSH.device)

        self.NodeGhoNeighbors = -1 * torch.ones(self.ValidGeoCorner.size(0), 3, 4, dtype=torch.int32,
                                                device=self.CornerSDF.device)

        self.NodeGhoCoeff = -1 * torch.ones(self.ValidGeoCorner.size(0), 3, 4, dtype=torch.float32,
                                            device=self.CornerSDF.device)

    def FindAllCornerPos(self, corner_pos, depth):
        num_node, num_innode, _ = corner_pos.shape
        node_len = 1.0 / (2 ** (depth + 1))
        corners_pos = torch.zeros(num_node, num_innode, 8, 3).to(self.CornerSH.device)
        zeros = torch.zeros_like(depth)

        corners_pos[:, :, 0] = corner_pos
        corners_pos[:, :, 1] = corner_pos + torch.unsqueeze(torch.stack((zeros, node_len, zeros), dim=1),
                                                            dim=1).repeat(1, num_innode, 1)
        corners_pos[:, :, 2] = corner_pos + torch.unsqueeze(torch.stack((node_len, zeros, zeros), dim=1),
                                                            dim=1).repeat(1, num_innode, 1)
        corners_pos[:, :, 3] = corner_pos + torch.unsqueeze(torch.stack((node_len, node_len, zeros), dim=1),
                                                            dim=1).repeat(1, num_innode, 1)
        corners_pos[:, :, 4] = corner_pos + torch.unsqueeze(torch.stack((zeros, zeros, node_len), dim=1),
                                                            dim=1).repeat(1, num_innode, 1)
        corners_pos[:, :, 5] = corner_pos + torch.unsqueeze(torch.stack((zeros, node_len, node_len), dim=1),
                                                            dim=1).repeat(1, num_innode, 1)
        corners_pos[:, :, 6] = corner_pos + torch.unsqueeze(torch.stack((node_len, zeros, node_len), dim=1),
                                                            dim=1).repeat(1, num_innode, 1)
        corners_pos[:, :, 7] = corner_pos + torch.unsqueeze(torch.stack((node_len, node_len, node_len), dim=1),
                                                            dim=1).repeat(1, num_innode, 1)

        return corners_pos

    def FindCornerIndex(self, corner_pos, depth):
        corners_pos = self.FindAllCornerPos(corner_pos, depth)

        scale = 2 ** (self.depth_limit + 1)

        chunk_size = 200000
        all_corners_indices = []
        for i in tqdm(range(0, corners_pos.size(0), chunk_size)):
            chunk_pos = corners_pos[i:i + chunk_size]
            corners_indices = torch.round(chunk_pos * scale)
            all_corners_indices.append(corners_indices)
        corners_full = torch.cat(all_corners_indices, dim=0)

        return corners_full

    def CalCorner(self, nodes):
        Q, _ = nodes.shape

        curr = nodes.clone()
        mask = torch.ones(Q, device=curr.device, dtype=torch.bool)
        output = torch.zeros(Q, 3, device=curr.device, dtype=self.data.dtype)

        while True:
            output[mask] += curr[:, 1:]
            output[mask] /= self.N

            good_mask = curr[:, 0] != 0
            if not good_mask.any():
                break
            mask[mask.clone()] = good_mask

            curr = self._unpack_index(self.parent_depth[curr[good_mask, 0], 0].long())

        return output

    def AllocateCornerMem(self, new_corner_pos, corners_pos, node_keys):
        unique_mask = self.RebuildCornerDict(new_corner_pos)

        corners_pos_l = corners_pos.view(corners_pos.shape[0] * 8, -1)
        corner_index = self.FastFromPosGetCornerIndex(corners_pos_l)

        node_corners = corner_index.view(-1, 8).type(torch.int32)

        self.NodeCorners[[node_keys[:, 0].long(), node_keys[:, 1].long(),
                          node_keys[:, 2].long(), node_keys[:, 3].long()]] = node_corners

        error_mask = (self.NodeCorners[:] == -1).nonzero()
        if error_mask.numel() > 0:
            a = error_mask
        error_mask = (self.NodeCorners[:] >= self.CornerSH.size(0)).nonzero()
        if error_mask.numel() > 0:
            a = error_mask

        return unique_mask

    def InitializeCorners(self):
        with torch.no_grad():
            corner_pos = torch.Tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
            scale = 2 ** (self.depth_limit + 1)
            corner_pos = torch.round(corner_pos * scale)
            self.RebuildCornerDict(corner_pos)

            corner_pos = torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
                                       [0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]])
            depth = torch.Tensor([0, 0, 0, 0,
                                  0, 0, 0, 0])

            corner_pos_us = torch.unsqueeze(corner_pos, dim=1)
            corners_indices = self.FindCornerIndex(corner_pos_us, depth)
            corners_indices = corners_indices.view([8, 8, 3])

            offest = torch.Tensor([0.1, 0.1, 0.1])
            corner_offest_pos = corner_pos + offest

            node_keys = self[LocalIndex(corner_offest_pos.to(self.CornerSH.device))].key
            node_keys = torch.stack(node_keys, dim=-1).to(device=self.data.device)

            corners_indices = torch.unsqueeze(corners_indices, dim=0)
            unique_new_ni = self.FindUniqueNodeIndices(corners_indices.shape[0], corners_indices)

            corners_indices = torch.squeeze(corners_indices, dim=0)
            self.AllocateCornerMem(unique_new_ni, corners_indices, node_keys)
            self.CornerD[:, ..., -1] = 1.0
            # self.CornerSDF[:, ..., -1] = 0.2
            # self.CornerSDF.data[:] = torch.randn(self.CornerSDF.size(0), 1)
            self.LearnS[:] = 1.0
            self.Beta[:] = 0.1

    def InitializeSDF(self):
        with torch.no_grad():
            reso = int(self.CornerSH.shape[0] ** (1 / 3))
            xx, yy, zz = torch.meshgrid(torch.arange(0, reso + 1, 1), torch.arange(0, reso + 1, 1),
                                        torch.arange(0, reso + 1, 1))
            reso_index = torch.stack((xx, yy, zz), dim=-1)
            reso_index = reso_index.view(-1, 3).to(self.CornerSH.device)

            grid_pos = reso_index / reso

            sdf = (((grid_pos[:, 0] - 0.5) ** 2 + (grid_pos[:, 1] - 0.5) ** 2 + (
                    grid_pos[:, 2] - 0.5) ** 2) ** 0.5 - 0.2) * 1.0

            reso_max = 2 ** (self.depth_limit + 1)
            reso_index[:, 0] = reso_index[:, 0] * (reso_max / reso)
            reso_index[:, 1] = reso_index[:, 1] * (reso_max / reso)
            reso_index[:, 2] = reso_index[:, 2] * (reso_max / reso)

            corner_index = self.FromPosGetCornerIndex(reso_index)

            self.CornerSDF.data[corner_index.long(), 0] = torch.tensor(sdf, dtype=torch.float32,
                                                                       device=self.CornerSH.device)

    def InitializeOctree(self):
        self.InitializeSDF()

        corners = torch.arange(0, self.CornerSDF.size(0), dtype=torch.int32, device=self.CornerSDF.device)
        self.SetGeoCorner(corner=corners)

        self.FindAllNeighCorner()

        self.FindGhostNeighbors()

        ksize = 5
        self.NodeGaussNeighbors, self.NodeGaussKernals = self.FindGaussianNeigh(ksize=ksize, sigma=0.8)

        ksize = 3
        self.NodeGaussGradNeighbors, self.NodeGaussGradKernals = self.FindCubicGaussianNeigh(sigma=0.8)
        self.NodeGaussGradKernals[:, int(ksize * ksize * ksize / 2)] = self.NodeGaussGradKernals[:,
                                                                       int(ksize * ksize * ksize / 2)] - 1.0

    def FindUniqueNodeIndices(self, num, new_nodes_indices):
        unique_new_ni = torch.zeros(num, 19, 3)
        unique_new_ni[:, 0] = new_nodes_indices[:, 0, 1]
        unique_new_ni[:, 1] = new_nodes_indices[:, 0, 2]
        unique_new_ni[:, 2] = new_nodes_indices[:, 0, 3]
        unique_new_ni[:, 3] = new_nodes_indices[:, 1, 3]
        unique_new_ni[:, 4] = new_nodes_indices[:, 2, 3]

        unique_new_ni[:, 5] = new_nodes_indices[:, 4, 0]
        unique_new_ni[:, 6] = new_nodes_indices[:, 4, 1]
        unique_new_ni[:, 7] = new_nodes_indices[:, 5, 1]
        unique_new_ni[:, 8] = new_nodes_indices[:, 4, 2]
        unique_new_ni[:, 9] = new_nodes_indices[:, 4, 3]
        unique_new_ni[:, 10] = new_nodes_indices[:, 5, 3]
        unique_new_ni[:, 11] = new_nodes_indices[:, 6, 2]
        unique_new_ni[:, 12] = new_nodes_indices[:, 6, 3]
        unique_new_ni[:, 13] = new_nodes_indices[:, 7, 3]

        unique_new_ni[:, 14] = new_nodes_indices[:, 4, 5]
        unique_new_ni[:, 15] = new_nodes_indices[:, 4, 6]
        unique_new_ni[:, 16] = new_nodes_indices[:, 4, 7]
        unique_new_ni[:, 17] = new_nodes_indices[:, 5, 7]
        unique_new_ni[:, 18] = new_nodes_indices[:, 6, 7]

        return unique_new_ni

    def CalNewNodeCornerData(self, num, ori_data):
        new_data = torch.zeros(num, 19, ori_data.shape[2])

        new_data[:, 0] = (ori_data[:, 0] + ori_data[:, 1]) / 2.0  # bottom_back_mid
        new_data[:, 1] = (ori_data[:, 0] + ori_data[:, 2]) / 2.0  # bottom_mid_left
        new_data[:, 3] = (ori_data[:, 1] + ori_data[:, 3]) / 2.0  # bottom_mid_right
        new_data[:, 2] = (new_data[:, 1] + new_data[:, 3]) / 2.0  # bottom_mid_mid
        new_data[:, 4] = (ori_data[:, 2] + ori_data[:, 3]) / 2.0  # bottom_front_mid

        new_data[:, 5] = (ori_data[:, 0] + ori_data[:, 4]) / 2.0  # mid_back_left
        new_data[:, 7] = (ori_data[:, 1] + ori_data[:, 5]) / 2.0  # mid_back_right
        new_data[:, 6] = (new_data[:, 5] + new_data[:, 7]) / 2.0  # mid_back_mid
        new_data[:, 11] = (ori_data[:, 2] + ori_data[:, 6]) / 2.0  # mid_front_left
        new_data[:, 13] = (ori_data[:, 3] + ori_data[:, 7]) / 2.0  # mid_front_right
        new_data[:, 12] = (new_data[:, 11] + new_data[:, 13]) / 2.0  # mid_front_mid
        new_data[:, 8] = (new_data[:, 5] + new_data[:, 11]) / 2.0  # mid_front_left
        new_data[:, 10] = (new_data[:, 7] + new_data[:, 13]) / 2.0  # mid_back_right
        new_data[:, 9] = (new_data[:, 8] + new_data[:, 10]) / 2.0  # mid_back_mid

        new_data[:, 14] = (ori_data[:, 4] + ori_data[:, 5]) / 2.0  # top_back_mid
        new_data[:, 15] = (ori_data[:, 4] + ori_data[:, 6]) / 2.0  # top_mid_left
        new_data[:, 17] = (ori_data[:, 5] + ori_data[:, 7]) / 2.0  # top_mid_right
        new_data[:, 16] = (new_data[:, 15] + new_data[:, 17]) / 2.0  # top_mid_mid
        new_data[:, 18] = (ori_data[:, 6] + ori_data[:, 7]) / 2.0  # top_front_mid

        return new_data

    def AllocateNewNodeCornerData(self, num, new_nodes_indices, unique_new_ni, unique_mask):
        with torch.no_grad():
            ori_ni = torch.zeros(num, 8, 3, device=self.CornerSH.device)
            ori_ni[:, 0] = new_nodes_indices[:, 0, 0]
            ori_ni[:, 1] = new_nodes_indices[:, 1, 1]
            ori_ni[:, 2] = new_nodes_indices[:, 2, 2]
            ori_ni[:, 3] = new_nodes_indices[:, 3, 3]
            ori_ni[:, 4] = new_nodes_indices[:, 4, 4]
            ori_ni[:, 5] = new_nodes_indices[:, 5, 5]
            ori_ni[:, 6] = new_nodes_indices[:, 6, 6]
            ori_ni[:, 7] = new_nodes_indices[:, 7, 7]
            ori_ni_l = ori_ni.view([num * 8, 3])

            corner_index = self.FastFromPosGetCornerIndex(ori_ni_l)

            ori_sdf = self.CornerSDF[corner_index.long()]
            ori_sh = self.CornerSH[corner_index.long()]

            ori_sdf = ori_sdf.view([num, 8, -1])
            ori_sh = ori_sh.view([num, 8, -1])

            new_sdf = self.CalNewNodeCornerData(num, ori_sdf).to(self.CornerSH.device)
            new_sh = self.CalNewNodeCornerData(num, ori_sh).to(self.CornerSH.device)

            unique_new_ni_l = unique_new_ni.view([num * 19, 3]).to(self.CornerSH.device)

            all_corners_indices = []
            chunk_size = 200000
            for i in tqdm(range(0, unique_new_ni_l.size(0), chunk_size)):
                corner_index = self.FastFromPosGetCornerIndex(unique_new_ni_l[i:i + chunk_size])
                all_corners_indices.append(corner_index)
            corner_index = torch.cat(all_corners_indices, dim=0)

            # corner_index = self.FromPosGetCornerIndex(unique_new_ni_l)
            corner_index = corner_index.view(-1)

            new_sdf = new_sdf.view(-1, 1)
            new_sh = new_sh.view(-1, 27)

            # corner_index = corner_index[unique_mask]
            # new_sdf = new_sdf[unique_mask]
            # new_sh = new_sh[unique_mask]

            self.CornerSDF[corner_index.long()] = new_sdf
            self.CornerSH[corner_index.long()] = new_sh

            # for i in range(num):
            #     # self.CornerD[indices_new_values[i].long()] = new_d[i]
            #     self.CornerSDF[corner_index[i].long()] = new_sdf[i]
            #     self.CornerSH[corner_index[i].long()] = new_sh[i]

    def GenerateNewCorners(self, node_keys, node_corners, node_depths, tri_inside=True):
        new_corners_pos = self.FindAllCornerPos(node_corners, node_depths)
        new_corners_pos = new_corners_pos.view([new_corners_pos.shape[0], 8, 3])

        new_nodes_indices = self.FindCornerIndex(new_corners_pos, node_depths)
        new_nodes_indices_l = new_nodes_indices.view(new_corners_pos.shape[0] * 8, 8, 3)

        unique_new_ni = self.FindUniqueNodeIndices(new_corners_pos.shape[0], new_nodes_indices)

        node_keys_l = node_keys.view(new_corners_pos.shape[0] * 8, -1)

        unique_mask = self.AllocateCornerMem(unique_new_ni, new_nodes_indices_l, node_keys_l)

        if tri_inside:
            self.AllocateNewNodeCornerData(new_corners_pos.shape[0], new_nodes_indices, unique_new_ni, unique_mask)

        return unique_new_ni

    def ResizeCapacity(self, num_nc):
        self.child = torch.cat((self.child,
                                torch.zeros((num_nc, *self.child.shape[1:]),
                                            dtype=self.child.dtype,
                                            device=self.CornerSH.device)))
        self.parent_depth = torch.cat((self.parent_depth,
                                       torch.zeros((num_nc, *self.parent_depth.shape[1:]),
                                                   dtype=self.parent_depth.dtype,
                                                   device=self.CornerSH.device)))
        self.NodeCorners = torch.cat((self.NodeCorners,
                                      torch.zeros((num_nc, *self.NodeCorners.shape[1:]),
                                                  dtype=self.NodeCorners.dtype,
                                                  device=self.CornerSH.device)))

    def RefineCorners(self, repeats=1, sel=None, tri_inside=True):
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        with torch.no_grad():
            all_new_ni = []
            for repeat_id in range(repeats):
                filled = self.n_internal
                if sel is None:
                    # Default all leaves
                    sel = (*self._all_leaves().T,)
                depths = self.parent_depth[sel[0], 1]
                # Filter by depth & leaves
                good_mask = (depths < self.depth_limit) & (self.child[sel] == 0)
                good_mask = good_mask.to(sel[0].device)
                sel = [t[good_mask] for t in sel]
                leaf_node = torch.stack(sel, dim=-1).to(device=self.data.device)
                num_nc = len(sel[0])
                if num_nc == 0:
                    # Nothing to do
                    return False
                new_filled = filled + num_nc

                # cap_needed = new_filled - self.capacity
                # if cap_needed > 0:
                #     self._resize_add_cap(cap_needed)
                #     resized = True
                self.ResizeCapacity(num_nc)

                new_idxs = torch.arange(filled, filled + num_nc,
                                        device=leaf_node.device, dtype=self.child.dtype)  # NNC

                self.child[filled:new_filled] = 0
                self.child[sel] = new_idxs - leaf_node[:, 0].to(torch.int32)
                # self.data.data[filled:new_filled] = self.data.data[
                # sel][:, None, None, None]
                self.parent_depth[filled:new_filled, 0] = self._pack_index(leaf_node)  # parent
                self.parent_depth[filled:new_filled, 1] = self.parent_depth[
                                                              leaf_node[:, 0], 1] + 1  # depth

                n_0 = torch.Tensor([0, 0, 1, 1, 0, 0, 1, 1])
                n_1 = torch.Tensor([0, 1, 0, 1, 0, 1, 0, 1])
                n_2 = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1])

                new_node_keys = torch.zeros(num_nc, 8, 4)
                new_node_keys[:, :, 0] = new_idxs.repeat(8, 1).T
                new_node_keys[:, :, 1] = n_0.repeat(num_nc, 1)
                new_node_keys[:, :, 2] = n_1.repeat(num_nc, 1)
                new_node_keys[:, :, 3] = n_2.repeat(num_nc, 1)

                node_corners = self.CalCorner(leaf_node)
                node_depth = depths + 1

                node_corners = torch.unsqueeze(node_corners, dim=1)
                new_ni = self.GenerateNewCorners(new_node_keys, node_corners, node_depth, tri_inside)
                all_new_ni.append(new_ni)

                # if repeat_id < repeats - 1:
                # Infer new selector
                t1 = torch.arange(filled, new_filled,
                                  device=self.data.device).repeat_interleave(self.N ** 3)
                rangen = torch.arange(self.N, device=self.data.device)
                t2 = rangen.repeat_interleave(self.N ** 2).repeat(
                    new_filled - filled)
                t3 = rangen.repeat_interleave(self.N).repeat(
                    (new_filled - filled) * self.N)
                t4 = rangen.repeat((new_filled - filled) * self.N ** 2)
                new_sel = (t1, t2, t3, t4)

            self._n_internal += num_nc
            self._c_depth = max(self._c_depth, torch.max(node_depth, dim=0)[0])
        if repeats > 0:
            self._invalidate()

        all_new_ni = torch.cat(all_new_ni, dim=0)
        return all_new_ni, new_sel

    def TrilinearInterpolation(self, sigma_d, rgb_d, pos, low_pos, node_length, only_sigma=False):
        node_length_L = node_length.repeat(3, 1).T

        pos_d = (pos - low_pos) / node_length_L

        wa, wb = 1.0 - pos_d, pos_d

        c00 = sigma_d[:, 0:1] * wa[:, 0:1] + sigma_d[:, 2:3] * wb[:, 0:1]
        c01 = sigma_d[:, 4:5] * wa[:, 0:1] + sigma_d[:, 6:7] * wb[:, 0:1]
        c10 = sigma_d[:, 1:2] * wa[:, 0:1] + sigma_d[:, 3:4] * wb[:, 0:1]
        c11 = sigma_d[:, 5:6] * wa[:, 0:1] + sigma_d[:, 7:] * wb[:, 0:1]
        c0 = c00 * wa[:, 1:2] + c10 * wb[:, 1:2]
        c1 = c01 * wa[:, 1:2] + c11 * wb[:, 1:2]
        sigma = c0 * wa[:, 2:] + c1 * wb[:, 2:]

        if (only_sigma):
            return sigma

        c00 = rgb_d[:, 0] * wa[:, 0:1] + rgb_d[:, 2] * wb[:, 0:1]
        c01 = rgb_d[:, 4] * wa[:, 0:1] + rgb_d[:, 6] * wb[:, 0:1]
        c10 = rgb_d[:, 1] * wa[:, 0:1] + rgb_d[:, 3] * wb[:, 0:1]
        c11 = rgb_d[:, 5] * wa[:, 0:1] + rgb_d[:, 7] * wb[:, 0:1]
        c0 = c00 * wa[:, 1:2] + c10 * wb[:, 1:2]
        c1 = c01 * wa[:, 1:2] + c11 * wb[:, 1:2]
        rgb = c0 * wa[:, 2:] + c1 * wb[:, 2:]

        return rgb, sigma, pos_d

    def TriInterpS(self, data, pos, low_pos, node_length):
        node_length_L = node_length.repeat(3, 1).T

        pos_d = (pos - low_pos) / node_length_L

        wa, wb = 1.0 - pos_d, pos_d

        c00 = data[:, 0:1] * wa[:, 0:1] + data[:, 2:3] * wb[:, 0:1]
        c01 = data[:, 4:5] * wa[:, 0:1] + data[:, 6:7] * wb[:, 0:1]
        c10 = data[:, 1:2] * wa[:, 0:1] + data[:, 3:4] * wb[:, 0:1]
        c11 = data[:, 5:6] * wa[:, 0:1] + data[:, 7:] * wb[:, 0:1]
        c0 = c00 * wa[:, 1:2] + c10 * wb[:, 1:2]
        c1 = c01 * wa[:, 1:2] + c11 * wb[:, 1:2]
        data_ti = c0 * wa[:, 2:] + c1 * wb[:, 2:]

        return data_ti

    def TriInterpSH(self, data, pos, low_pos, node_length):
        node_length_L = node_length.repeat(3, 1).T

        pos_d = (pos - low_pos) / node_length_L

        wa, wb = 1.0 - pos_d, pos_d

        c00 = data[:, 0] * wa[:, 0:1] + data[:, 2] * wb[:, 0:1]
        c01 = data[:, 4] * wa[:, 0:1] + data[:, 6] * wb[:, 0:1]
        c10 = data[:, 1] * wa[:, 0:1] + data[:, 3] * wb[:, 0:1]
        c11 = data[:, 5] * wa[:, 0:1] + data[:, 7] * wb[:, 0:1]
        c0 = c00 * wa[:, 1:2] + c10 * wb[:, 1:2]
        c1 = c01 * wa[:, 1:2] + c11 * wb[:, 1:2]
        rgb = c0 * wa[:, 2:] + c1 * wb[:, 2:]

        return rgb

    def CalSigma(self, points):
        with torch.no_grad():
            treeview = self[LocalIndex(points)]
            leaf_node = torch.stack(treeview.key, dim=-1)

            node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
            leaf_node[:, 2], leaf_node[:, 3]]
            node_num = node_corners.shape[0]
            node_corners = node_corners.view([node_num * 8, 1])
            nc_d = self.CornerD[node_corners.long()]
            nc_d = nc_d.view([node_num, 8, -1])
            nc_sh = self.CornerSH[node_corners.long()]
            nc_sh = nc_sh.view([node_num, 8, -1])

            sigma = self.TrilinearInterpolation(nc_d, nc_sh, points, treeview, only_sigma=True)
            return sigma

    def SigmaEvaluaion(self, leaf_mask, sigma_threshold):
        # leaf_mask = self.depths.cpu() == self.max_depth
        leaf_ind = torch.where(leaf_mask)[0]
        del leaf_mask

        sample_num = 10
        chunk_size = 2000 // sample_num

        all_sigma = []
        for i in tqdm(range(0, leaf_ind.size(0), chunk_size)):
            chunk_inds = leaf_ind[i:i + chunk_size]
            points = self[chunk_inds].sample_local(sample_num)
            points = points.view(-1, 3)

            sigma = self.CalSigma(points)
            sigma = sigma.view(-1, sample_num, 1).mean(dim=1)
            all_sigma.append(sigma)

        all_sigma_data = torch.cat(all_sigma, dim=0)

        sigma_mask = all_sigma_data > sigma_threshold
        return sigma_mask

    def LOTPartial(self, data_sel=None, data_format=None, dtype=None, device=None):
        if device is None:
            device = self.data.device
        if data_sel is None:
            new_data_dim = self.data_dim
            sel_indices = None
        else:
            sel_indices = torch.arange(self.data_dim)[data_sel]
            if sel_indices.ndim == 0:
                sel_indices = sel_indices.unsqueeze(0)
            new_data_dim = sel_indices.numel()
        if dtype is None:
            dtype = self.data.dtype
        t2 = LOctreeA(N=self.N, data_dim=new_data_dim,
                      data_format=data_format or str(self.data_format),
                      depth_limit=self.depth_limit,
                      geom_resize_fact=self.geom_resize_fact,
                      dtype=dtype,
                      device=device)

        def copy_to_device(x):
            return torch.empty(x.shape, dtype=x.dtype, device=device).copy_(x)

        t2.invradius = copy_to_device(self.invradius)
        t2.offset = copy_to_device(self.offset)
        t2.child = copy_to_device(self.child)
        t2.parent_depth = copy_to_device(self.parent_depth)

        t2.NodeCorners = copy_to_device(self.NodeCorners)
        t2.CornerIndex = copy_to_device(self.CornerIndex)
        t2.CornerD = nn.Parameter(copy_to_device(self.CornerD.data))
        t2.CornerSH = nn.Parameter(copy_to_device(self.CornerSH.data))

        t2._n_internal = copy_to_device(self._n_internal)
        t2._n_free = copy_to_device(self._n_free)
        t2._n_corners = copy_to_device(self._n_corners)

        if self.extra_data is not None:
            t2.extra_data = copy_to_device(self.extra_data)
        else:
            t2.extra_data = None
        t2.data_format = self.data_format
        if data_sel is None:
            t2.data = nn.Parameter(copy_to_device(self.data.data))
        else:
            t2.data = nn.Parameter(copy_to_device(self.data.data[..., sel_indices].contiguous()))
        return t2

    def FindNeighCornerA(self):
        sel = (*self._all_leaves().T,)
        leaf_node = torch.stack(sel, dim=-1).to(device=self.data.device)
        depths = self.parent_depth[sel[0], 1]

        node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]]

        node_corners_0 = node_corners[:, 0].long()
        node_corners_0_x = node_corners[:, 2]
        node_corners_0_y = node_corners[:, 1]
        node_corners_0_z = node_corners[:, 4]

        self.NodeNeighbors[node_corners_0, 0] = node_corners_0_x
        self.NodeNeighbors[node_corners_0, 1] = node_corners_0_y
        self.NodeNeighbors[node_corners_0, 2] = node_corners_0_z
        self.NodeNeighbors[node_corners_0, 3] = depths
        self.NodeNeighbors[node_corners_0, 4] = depths
        self.NodeNeighbors[node_corners_0, 5] = depths
        """"""""""""
        """"""""""""
        node_corners_1 = node_corners[:, 1].long()
        node_corners_1_x = node_corners[:, 3]
        node_corners_1_z = node_corners[:, 5]

        diff = self.NodeNeighbors[node_corners_1, 3] - depths
        refill_index = (diff < 0)
        node_corners_1_ri = node_corners_1[refill_index]
        self.NodeNeighbors[node_corners_1_ri, 0] = node_corners_1_x[refill_index]
        self.NodeNeighbors[node_corners_1_ri, 3] = depths[refill_index]

        diff = self.NodeNeighbors[node_corners_1, 5] - depths
        refill_index = (diff < 0)
        node_corners_1_ri = node_corners_1[refill_index]
        self.NodeNeighbors[node_corners_1_ri, 2] = node_corners_1_z[refill_index]
        self.NodeNeighbors[node_corners_1_ri, 5] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_2 = node_corners[:, 2].long()
        node_corners_2_y = node_corners[:, 3]
        node_corners_2_z = node_corners[:, 6]

        diff = self.NodeNeighbors[node_corners_2, 4] - depths
        refill_index = (diff < 0)
        node_corners_2_ri = node_corners_2[refill_index]
        self.NodeNeighbors[node_corners_2_ri, 1] = node_corners_2_y[refill_index]
        self.NodeNeighbors[node_corners_2_ri, 4] = depths[refill_index]

        diff = self.NodeNeighbors[node_corners_2, 5] - depths
        refill_index = (diff < 0)
        node_corners_2_ri = node_corners_2[refill_index]
        self.NodeNeighbors[node_corners_2_ri, 2] = node_corners_2_z[refill_index]
        self.NodeNeighbors[node_corners_2_ri, 5] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_3 = node_corners[:, 3].long()
        node_corners_3_z = node_corners[:, 7]

        diff = self.NodeNeighbors[node_corners_3, 5] - depths
        refill_index = (diff < 0)
        node_corners_3_ri = node_corners_3[refill_index]
        self.NodeNeighbors[node_corners_3_ri, 2] = node_corners_3_z[refill_index]
        self.NodeNeighbors[node_corners_3_ri, 5] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_4 = node_corners[:, 4].long()
        node_corners_4_x = node_corners[:, 6]
        node_corners_4_y = node_corners[:, 5]

        diff = self.NodeNeighbors[node_corners_4, 3] - depths
        refill_index = (diff < 0)
        node_corners_4_ri = node_corners_4[refill_index]
        self.NodeNeighbors[node_corners_4_ri, 0] = node_corners_4_x[refill_index]
        self.NodeNeighbors[node_corners_4_ri, 3] = depths[refill_index]

        diff = self.NodeNeighbors[node_corners_4, 4] - depths
        refill_index = (diff < 0)
        node_corners_4_ri = node_corners_4[refill_index]
        self.NodeNeighbors[node_corners_4_ri, 1] = node_corners_4_y[refill_index]
        self.NodeNeighbors[node_corners_4_ri, 4] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_5 = node_corners[:, 5].long()
        node_corners_5_x = node_corners[:, 7]

        diff = self.NodeNeighbors[node_corners_5, 3] - depths
        refill_index = (diff < 0)
        node_corners_5_ri = node_corners_5[refill_index]
        self.NodeNeighbors[node_corners_5_ri, 0] = node_corners_5_x[refill_index]
        self.NodeNeighbors[node_corners_5_ri, 3] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_6 = node_corners[:, 6].long()
        node_corners_6_y = node_corners[:, 7]

        diff = self.NodeNeighbors[node_corners_6, 4] - depths
        refill_index = (diff < 0)
        node_corners_6_ri = node_corners_6[refill_index]
        self.NodeNeighbors[node_corners_6_ri, 1] = node_corners_6_y[refill_index]
        self.NodeNeighbors[node_corners_6_ri, 4] = depths[refill_index]

    def FindAllNeighCorner(self):
        all_node_neigbors = -1 * torch.ones(self.CornerSH.size(0), 12, dtype=torch.int32,
                                            device=self.CornerSH.device)

        sel = (*self._all_leaves().T,)
        leaf_node = torch.stack(sel, dim=-1).to(device=self.data.device)
        depths = self.parent_depth[sel[0], 1]

        node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]]

        node_corners_0 = node_corners[:, 0].long()
        node_corners_0_x_1 = node_corners[:, 2]
        node_corners_0_y_1 = node_corners[:, 1]
        node_corners_0_z_1 = node_corners[:, 4]

        all_node_neigbors[node_corners_0, 1] = node_corners_0_x_1
        all_node_neigbors[node_corners_0, 3] = node_corners_0_y_1
        all_node_neigbors[node_corners_0, 5] = node_corners_0_z_1
        all_node_neigbors[node_corners_0, 7] = depths
        all_node_neigbors[node_corners_0, 9] = depths
        all_node_neigbors[node_corners_0, 11] = depths
        """"""""""""
        """"""""""""
        node_corners_1 = node_corners[:, 1].long()
        node_corners_1_x_1 = node_corners[:, 3]
        node_corners_1_y_0 = node_corners[:, 0]
        node_corners_1_z_1 = node_corners[:, 5]

        diff = all_node_neigbors[node_corners_1, 7] - depths
        refill_index = (diff < 0)
        node_corners_1_ri = node_corners_1[refill_index]
        all_node_neigbors[node_corners_1_ri, 1] = node_corners_1_x_1[refill_index]
        all_node_neigbors[node_corners_1_ri, 7] = depths[refill_index]

        diff = all_node_neigbors[node_corners_1, 8] - depths
        refill_index = (diff < 0)
        node_corners_1_ri = node_corners_1[refill_index]
        all_node_neigbors[node_corners_1_ri, 2] = node_corners_1_y_0[refill_index]
        all_node_neigbors[node_corners_1_ri, 8] = depths[refill_index]

        diff = all_node_neigbors[node_corners_1, 11] - depths
        refill_index = (diff < 0)
        node_corners_1_ri = node_corners_1[refill_index]
        all_node_neigbors[node_corners_1_ri, 5] = node_corners_1_z_1[refill_index]
        all_node_neigbors[node_corners_1_ri, 11] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_2 = node_corners[:, 2].long()
        node_corners_2_x_0 = node_corners[:, 0]
        node_corners_2_y_1 = node_corners[:, 3]
        node_corners_2_z_1 = node_corners[:, 6]

        diff = all_node_neigbors[node_corners_2, 6] - depths
        refill_index = (diff < 0)
        node_corners_2_ri = node_corners_2[refill_index]
        all_node_neigbors[node_corners_2_ri, 0] = node_corners_2_x_0[refill_index]
        all_node_neigbors[node_corners_2_ri, 6] = depths[refill_index]

        diff = all_node_neigbors[node_corners_2, 9] - depths
        refill_index = (diff < 0)
        node_corners_2_ri = node_corners_2[refill_index]
        all_node_neigbors[node_corners_2_ri, 3] = node_corners_2_y_1[refill_index]
        all_node_neigbors[node_corners_2_ri, 9] = depths[refill_index]

        diff = all_node_neigbors[node_corners_2, 11] - depths
        refill_index = (diff < 0)
        node_corners_2_ri = node_corners_2[refill_index]
        all_node_neigbors[node_corners_2_ri, 5] = node_corners_2_z_1[refill_index]
        all_node_neigbors[node_corners_2_ri, 11] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_3 = node_corners[:, 3].long()
        node_corners_3_x_0 = node_corners[:, 1]
        node_corners_3_y_0 = node_corners[:, 2]
        node_corners_3_z_1 = node_corners[:, 7]

        diff = all_node_neigbors[node_corners_3, 6] - depths
        refill_index = (diff < 0)
        node_corners_3_ri = node_corners_3[refill_index]
        all_node_neigbors[node_corners_3_ri, 0] = node_corners_3_x_0[refill_index]
        all_node_neigbors[node_corners_3_ri, 6] = depths[refill_index]

        diff = all_node_neigbors[node_corners_3, 8] - depths
        refill_index = (diff < 0)
        node_corners_3_ri = node_corners_3[refill_index]
        all_node_neigbors[node_corners_3_ri, 2] = node_corners_3_y_0[refill_index]
        all_node_neigbors[node_corners_3_ri, 8] = depths[refill_index]

        diff = all_node_neigbors[node_corners_3, 11] - depths
        refill_index = (diff < 0)
        node_corners_3_ri = node_corners_3[refill_index]
        all_node_neigbors[node_corners_3_ri, 5] = node_corners_3_z_1[refill_index]
        all_node_neigbors[node_corners_3_ri, 11] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_4 = node_corners[:, 4].long()
        node_corners_4_x_1 = node_corners[:, 6]
        node_corners_4_y_1 = node_corners[:, 5]
        node_corners_4_z_0 = node_corners[:, 0]

        diff = all_node_neigbors[node_corners_4, 7] - depths
        refill_index = (diff < 0)
        node_corners_4_ri = node_corners_4[refill_index]
        all_node_neigbors[node_corners_4_ri, 1] = node_corners_4_x_1[refill_index]
        all_node_neigbors[node_corners_4_ri, 7] = depths[refill_index]

        diff = all_node_neigbors[node_corners_4, 9] - depths
        refill_index = (diff < 0)
        node_corners_4_ri = node_corners_4[refill_index]
        all_node_neigbors[node_corners_4_ri, 3] = node_corners_4_y_1[refill_index]
        all_node_neigbors[node_corners_4_ri, 9] = depths[refill_index]

        diff = all_node_neigbors[node_corners_4, 10] - depths
        refill_index = (diff < 0)
        node_corners_4_ri = node_corners_4[refill_index]
        all_node_neigbors[node_corners_4_ri, 4] = node_corners_4_z_0[refill_index]
        all_node_neigbors[node_corners_4_ri, 10] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_5 = node_corners[:, 5].long()
        node_corners_5_x_1 = node_corners[:, 7]
        node_corners_5_y_0 = node_corners[:, 4]
        node_corners_5_z_0 = node_corners[:, 1]

        diff = all_node_neigbors[node_corners_5, 7] - depths
        refill_index = (diff < 0)
        node_corners_5_ri = node_corners_5[refill_index]
        all_node_neigbors[node_corners_5_ri, 1] = node_corners_5_x_1[refill_index]
        all_node_neigbors[node_corners_5_ri, 7] = depths[refill_index]

        diff = all_node_neigbors[node_corners_5, 8] - depths
        refill_index = (diff < 0)
        node_corners_5_ri = node_corners_5[refill_index]
        all_node_neigbors[node_corners_5_ri, 2] = node_corners_5_y_0[refill_index]
        all_node_neigbors[node_corners_5_ri, 8] = depths[refill_index]

        diff = all_node_neigbors[node_corners_5, 10] - depths
        refill_index = (diff < 0)
        node_corners_5_ri = node_corners_5[refill_index]
        all_node_neigbors[node_corners_5_ri, 4] = node_corners_5_z_0[refill_index]
        all_node_neigbors[node_corners_5_ri, 10] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_6 = node_corners[:, 6].long()
        node_corners_6_x_0 = node_corners[:, 4]
        node_corners_6_y_1 = node_corners[:, 7]
        node_corners_6_z_0 = node_corners[:, 2]

        diff = all_node_neigbors[node_corners_6, 6] - depths
        refill_index = (diff < 0)
        node_corners_6_ri = node_corners_6[refill_index]
        all_node_neigbors[node_corners_6_ri, 0] = node_corners_6_x_0[refill_index]
        all_node_neigbors[node_corners_6_ri, 6] = depths[refill_index]

        diff = all_node_neigbors[node_corners_6, 9] - depths
        refill_index = (diff < 0)
        node_corners_6_ri = node_corners_6[refill_index]
        all_node_neigbors[node_corners_6_ri, 3] = node_corners_6_y_1[refill_index]
        all_node_neigbors[node_corners_6_ri, 9] = depths[refill_index]

        diff = all_node_neigbors[node_corners_6, 10] - depths
        refill_index = (diff < 0)
        node_corners_6_ri = node_corners_6[refill_index]
        all_node_neigbors[node_corners_6_ri, 4] = node_corners_6_z_0[refill_index]
        all_node_neigbors[node_corners_6_ri, 10] = depths[refill_index]
        """"""""""""
        """"""""""""
        node_corners_7 = node_corners[:, 7].long()
        node_corners_7_x_0 = node_corners[:, 5]
        node_corners_7_y_0 = node_corners[:, 6]
        node_corners_7_z_0 = node_corners[:, 3]

        diff = all_node_neigbors[node_corners_7, 6] - depths
        refill_index = (diff < 0)
        node_corners_7_ri = node_corners_7[refill_index]
        all_node_neigbors[node_corners_7_ri, 0] = node_corners_7_x_0[refill_index]
        all_node_neigbors[node_corners_7_ri, 6] = depths[refill_index]

        diff = all_node_neigbors[node_corners_7, 8] - depths
        refill_index = (diff < 0)
        node_corners_7_ri = node_corners_7[refill_index]
        all_node_neigbors[node_corners_7_ri, 2] = node_corners_7_y_0[refill_index]
        all_node_neigbors[node_corners_7_ri, 8] = depths[refill_index]

        diff = all_node_neigbors[node_corners_7, 10] - depths
        refill_index = (diff < 0)
        node_corners_7_ri = node_corners_7[refill_index]
        all_node_neigbors[node_corners_7_ri, 4] = node_corners_7_z_0[refill_index]
        all_node_neigbors[node_corners_7_ri, 10] = depths[refill_index]

        self.NodeAllNeighLen = 1.0 / (2.0 ** (all_node_neigbors[self.ValidGeoCorner.long(), 6:] + 1))
        self.NodeAllNeighbors = torch.tensor(all_node_neigbors[self.ValidGeoCorner.long(), 0:6], dtype=torch.int32)

    def FindGhoNeighCore(self, corner_node_list, corner_node_list_2_0, corner_node_list_2_1, corner_index, corner_pos,
                         gho_node_neigh_len, dir_index):
        node_id = corner_node_list[corner_index]

        node_parent_id = self.parent_depth[node_id[:, 0].long(), 0]
        node_parent_id = self._unpack_index(node_parent_id).long()

        if dir_index == 0:
            node_dir_index = torch.tensor([0, 2], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 1:
            node_dir_index = torch.tensor([2, 0], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 2:
            node_dir_index = torch.tensor([0, 1], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 3:
            node_dir_index = torch.tensor([1, 0], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 4:
            node_dir_index = torch.tensor([0, 4], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 5:
            node_dir_index = torch.tensor([4, 0], dtype=torch.int32, device=self.CornerSH.device)

        finded_node = -1 * torch.ones(corner_index.size(0), 4, dtype=torch.int32, device=self.CornerSH.device)
        while True:
            corner_0_index = self.NodeCorners[[node_parent_id[:, 0], node_parent_id[:, 1],
                                               node_parent_id[:, 2], node_parent_id[:, 3]]][:, node_dir_index[0]]
            corner_2_node_index_0 = corner_node_list_2_0[corner_0_index.long(), node_dir_index[1]]
            corner_2_node_index_1 = corner_node_list_2_1[corner_0_index.long(), node_dir_index[1]]
            finded_parent_mask = (corner_2_node_index_0[:, 0] != -1)
            # node_dir_index = ((dir_index + 1) // 2) + ((dir_index + 1) % 2)
            # finded_parent_mask = (node_parent_id[:, node_dir_index] == ((dir_index + 1) % 2))
            finded_node_index = finded_parent_mask.nonzero().squeeze(1)
            finded_node[finded_node_index, 0] = corner_2_node_index_0[finded_parent_mask, 0]
            finded_node[finded_node_index, 1:4] = corner_2_node_index_1[finded_parent_mask, 0:3].type(torch.int32)
            # finded_node[finded_node_index, node_dir_index] = (dir_index % 2)

            unfinded_parent_index = (finded_node[:, 0] == -1).nonzero().squeeze(1)
            if unfinded_parent_index.size(0) != 0:
                # node_parent_id[:, node_dir_index] = -1
                node_parent_id[unfinded_parent_index] = self._unpack_index(self.parent_depth[
                                                                               node_parent_id[
                                                                                   unfinded_parent_index, 0].long(), 0].long())
            else:
                break

        if dir_index == 0:
            corner_dir_index = torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 1:
            corner_dir_index = torch.tensor([2, 3, 6, 7], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 2:
            corner_dir_index = torch.tensor([2, 0, 6, 4], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 3:
            corner_dir_index = torch.tensor([3, 1, 7, 5], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 4:
            corner_dir_index = torch.tensor([2, 3, 0, 1], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 5:
            corner_dir_index = torch.tensor([6, 7, 4, 5], dtype=torch.int32, device=self.CornerSH.device)

        finded_node = finded_node.long()

        finded_node_cor_0 = self.NodeCorners[
                                [finded_node[:, 0], finded_node[:, 1],
                                 finded_node[:, 2], finded_node[:, 3]]][:, corner_dir_index[0]]
        finded_node_cor_1 = self.NodeCorners[
                                [finded_node[:, 0], finded_node[:, 1],
                                 finded_node[:, 2], finded_node[:, 3]]][:, corner_dir_index[1]]
        finded_node_cor_2 = self.NodeCorners[
                                [finded_node[:, 0], finded_node[:, 1],
                                 finded_node[:, 2], finded_node[:, 3]]][:, corner_dir_index[2]]
        finded_node_cor_3 = self.NodeCorners[
                                [finded_node[:, 0], finded_node[:, 1],
                                 finded_node[:, 2], finded_node[:, 3]]][:, corner_dir_index[3]]

        node_length = 1.0 / (2 ** (self.parent_depth[finded_node[:, 0], 1] + 1))
        node_corner_pos = torch.zeros(finded_node.size(0), 8, 3, dtype=torch.float32,
                                      device=self.CornerSH.device)
        node_corner_pos[:, 0] = self.CalCorner(finded_node)
        node_corner_pos[:, 1] = node_corner_pos[:, 0]
        node_corner_pos[:, 1, 1] = node_corner_pos[:, 1, 1] + node_length
        node_corner_pos[:, 2] = node_corner_pos[:, 0]
        node_corner_pos[:, 2, 0] = node_corner_pos[:, 2, 0] + node_length
        node_corner_pos[:, 3] = node_corner_pos[:, 2]
        node_corner_pos[:, 3, 1] = node_corner_pos[:, 3, 1] + node_length
        node_corner_pos[:, 4] = node_corner_pos[:, 0]
        node_corner_pos[:, 4, 2] = node_corner_pos[:, 4, 2] + node_length
        node_corner_pos[:, 5] = node_corner_pos[:, 4]
        node_corner_pos[:, 5, 1] = node_corner_pos[:, 5, 1] + node_length
        node_corner_pos[:, 6] = node_corner_pos[:, 4]
        node_corner_pos[:, 6, 0] = node_corner_pos[:, 6, 0] + node_length
        node_corner_pos[:, 7] = node_corner_pos[:, 6]
        node_corner_pos[:, 7, 1] = node_corner_pos[:, 7, 1] + node_length

        all_corner_pos = torch.zeros(finded_node.size(0), 4, 3, dtype=torch.float32,
                                     device=self.CornerSH.device)
        all_corner_pos[:, 0] = node_corner_pos[:, corner_dir_index[0]]
        all_corner_pos[:, 1] = node_corner_pos[:, corner_dir_index[1]]
        all_corner_pos[:, 2] = node_corner_pos[:, corner_dir_index[2]]
        all_corner_pos[:, 3] = node_corner_pos[:, corner_dir_index[3]]

        all_node_cor = torch.zeros(finded_node.size(0), 4, dtype=torch.int32,
                                   device=self.CornerSH.device)
        all_node_cor[:, 0] = finded_node_cor_0
        all_node_cor[:, 1] = finded_node_cor_1
        all_node_cor[:, 2] = finded_node_cor_2
        all_node_cor[:, 3] = finded_node_cor_3

        corner_pos = corner_pos.view(finded_node.size(0), 1, 3)
        corner_pos = corner_pos.repeat(1, 4, 1).to(self.CornerSH.device)

        finded_corner_bool = torch.zeros(corner_index.size(0), device=self.CornerSH.device)
        finded_corner_bool = finded_corner_bool.type(torch.bool)

        if dir_index == 0 or dir_index == 1:
            pos_dir_index = torch.tensor([1, 2, 0], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 2 or dir_index == 3:
            pos_dir_index = torch.tensor([0, 2, 1], dtype=torch.int32, device=self.CornerSH.device)
        elif dir_index == 4 or dir_index == 5:
            pos_dir_index = torch.tensor([1, 0, 2], dtype=torch.int32, device=self.CornerSH.device)

        gho_dir_index = int(dir_index / 2)

        corner_pos_diff_index = (
                corner_pos[:, :, pos_dir_index[0]] - all_corner_pos[:, :, pos_dir_index[0]] == 0).nonzero()
        corner_pos_diff_index = corner_pos_diff_index.view(-1, 2, 2)
        if corner_pos_diff_index.size(0) > 0:
            node_cor_index = all_node_cor[
                [corner_pos_diff_index[:, :, 0].long(), corner_pos_diff_index[:, :, 1].long()]]

            cor_map_index = self.CornerMap[corner_index[corner_pos_diff_index[:, 0, 0]].long()].long()
            self.NodeGhoNeighbors[cor_map_index.long(), gho_dir_index, 0:2] = node_cor_index
            # self.NodeGhoNeighbors[corner_index[corner_pos_diff_index[:, 0, 0].long()], gho_dir_index, 2] = \
            #     pos_dir_index[1]

            diff_length = corner_pos[corner_pos_diff_index[:, :, 0].long(),
            corner_pos_diff_index[:, :, 1].long(), pos_dir_index[1]] - \
                          all_corner_pos[corner_pos_diff_index[:, :, 0].long(),
                          corner_pos_diff_index[:, :, 1].long(), pos_dir_index[1]]
            diff_length = torch.abs(diff_length)
            gho_node_neigh_len[cor_map_index, gho_dir_index, 0:2] = diff_length.to('cpu')

            diff_length = corner_pos[corner_pos_diff_index[:, 0, 0].long(),
            corner_pos_diff_index[:, 0, 1].long(), pos_dir_index[2]] - all_corner_pos[
                              corner_pos_diff_index[:, 0, 0].long(), corner_pos_diff_index[:, 0, 1].long(),
                              pos_dir_index[2]]
            diff_length = torch.abs(diff_length)
            self.NodeAllNeighLen[cor_map_index, dir_index] = diff_length

            finded_corner_bool[corner_pos_diff_index[:, 0, 0].long()] = True

        corner_pos_diff_index = (
                corner_pos[:, :, pos_dir_index[1]] - all_corner_pos[:, :, pos_dir_index[1]] == 0).nonzero()
        corner_pos_diff_index = corner_pos_diff_index.view(-1, 2, 2)
        if corner_pos_diff_index.size(0) > 0:
            node_cor_index = all_node_cor[
                [corner_pos_diff_index[:, :, 0].long(), corner_pos_diff_index[:, :, 1].long()]]

            cor_map_index = self.CornerMap[corner_index[corner_pos_diff_index[:, 0, 0].long()]].long()
            self.NodeGhoNeighbors[cor_map_index, gho_dir_index, 0:2] = node_cor_index
            # self.NodeGhoNeighbors[corner_index[corner_pos_diff_index[:, 0, 0].long()], gho_dir_index, 2] = \
            #     pos_dir_index[0]

            diff_length = corner_pos[corner_pos_diff_index[:, :, 0].long(),
            corner_pos_diff_index[:, :, 1].long(), pos_dir_index[0]] - \
                          all_corner_pos[corner_pos_diff_index[:, :, 0].long(),
                          corner_pos_diff_index[:, :, 1].long(), pos_dir_index[0]]
            diff_length = torch.abs(diff_length)
            gho_node_neigh_len[cor_map_index, gho_dir_index, 0:2] = diff_length.to('cpu')

            diff_length = corner_pos[corner_pos_diff_index[:, 0, 0].long(),
            corner_pos_diff_index[:, 0, 1].long(), pos_dir_index[2]] - all_corner_pos[
                              corner_pos_diff_index[:, 0, 0].long(), corner_pos_diff_index[:, 0, 1].long(),
                              pos_dir_index[2]]
            diff_length = torch.abs(diff_length)
            self.NodeAllNeighLen[cor_map_index, dir_index] = diff_length

            finded_corner_bool[corner_pos_diff_index[:, 0, 0].long()] = True

        finded_corner_b = corner_index[~finded_corner_bool]
        finded_node_cor_b = all_node_cor[~finded_corner_bool]
        finded_cor_pos = corner_pos[~finded_corner_bool]
        finded_all_cor_pos_b = all_corner_pos[~finded_corner_bool]

        cor_map_index = self.CornerMap[finded_corner_b.long()].long()
        self.NodeGhoNeighbors[cor_map_index, gho_dir_index] = finded_node_cor_b
        lenb = finded_all_cor_pos_b[:, 0, pos_dir_index[0]] - finded_cor_pos[:, 0, pos_dir_index[0]]
        lenb = torch.abs(lenb)
        gho_node_neigh_len[cor_map_index, gho_dir_index, 0] = lenb.to('cpu')
        lenb = finded_all_cor_pos_b[:, 1, pos_dir_index[0]] - finded_cor_pos[:, 1, pos_dir_index[0]]
        lenb = torch.abs(lenb)
        gho_node_neigh_len[cor_map_index, gho_dir_index, 1] = lenb.to('cpu')
        lenb = finded_all_cor_pos_b[:, 0, pos_dir_index[1]] - finded_cor_pos[:, 0, pos_dir_index[1]]
        lenb = torch.abs(lenb)
        gho_node_neigh_len[cor_map_index, gho_dir_index, 2] = lenb.to('cpu')
        lenb = finded_all_cor_pos_b[:, 2, pos_dir_index[1]] - finded_cor_pos[:, 2, pos_dir_index[1]]
        lenb = torch.abs(lenb)
        gho_node_neigh_len[cor_map_index, gho_dir_index, 3] = lenb.to('cpu')

        lenb = finded_all_cor_pos_b[:, 0, pos_dir_index[2]] - finded_cor_pos[:, 0, pos_dir_index[2]]
        lenb = torch.abs(lenb)
        self.NodeAllNeighLen[cor_map_index, dir_index] = lenb

    def CalGhostCoeff(self, corner_index, gho_node_neigh_len):
        ghost_corner_a_mask_1 = (
                self.NodeGhoNeighbors[self.CornerMap[corner_index].long(), :, 1] >= 0)
        ghost_corner_a_mask_2 = (
                self.NodeGhoNeighbors[self.CornerMap[corner_index].long(), :, 3] == -1)
        ghost_corner_a_mask = ghost_corner_a_mask_1 * ghost_corner_a_mask_2
        ghost_corner_a_index = torch.nonzero(ghost_corner_a_mask)

        cor_map_index = self.CornerMap[corner_index[ghost_corner_a_index[:, 0]].long()].long()
        gho_cor_a_len = gho_node_neigh_len[cor_map_index, ghost_corner_a_index[:, 1]]
        coeff = gho_cor_a_len[:, 0] / (gho_cor_a_len[:, 0] + gho_cor_a_len[:, 1])
        self.NodeGhoCoeff[cor_map_index, ghost_corner_a_index[:, 1], 0] = coeff.to(self.CornerSH.device)
        coeff = gho_cor_a_len[:, 1] / (gho_cor_a_len[:, 0] + gho_cor_a_len[:, 1])
        self.NodeGhoCoeff[cor_map_index, ghost_corner_a_index[:, 1], 1] = coeff.to(self.CornerSH.device)
        # gho_node_neigh_len[cor_map_index, ghost_corner_a_index[:, 1], 0] = gho_node_neigh_len[
        #                                                                        cor_map_index, ghost_corner_a_index[:,
        #                                                                                       1], 0] / (
        #                                                                            gho_node_neigh_len[
        #                                                                                cor_map_index, ghost_corner_a_index[
        #                                                                                               :, 1], 0] +
        #                                                                            gho_node_neigh_len[
        #                                                                                cor_map_index, ghost_corner_a_index[
        #                                                                                               :, 1], 1])
        # gho_node_neigh_len[cor_map_index, ghost_corner_a_index[:, 1], 1] = gho_node_neigh_len[
        #                                                                        cor_map_index, ghost_corner_a_index[:,
        #                                                                                       1], 1] / (
        #                                                                            gho_node_neigh_len[
        #                                                                                cor_map_index, ghost_corner_a_index[
        #                                                                                               :, 1], 0] +
        #                                                                            gho_node_neigh_len[
        #                                                                                cor_map_index, ghost_corner_a_index[
        #                                                                                               :, 1], 1])

        # len0 = self.NodeAllNeighLen[corner_index[ghost_corner_a_index[:, 0]],
        # (self.NodeGhoNeighbors[corner_index[ghost_corner_a_index[:, 0]], ghost_corner_a_index[:, 1], 2] * 2).long()]
        # len1 = self.NodeAllNeighLen[corner_index[ghost_corner_a_index[:, 0]],
        # (self.NodeGhoNeighbors[corner_index[ghost_corner_a_index[:, 0]], ghost_corner_a_index[:, 1], 2] * 2 + 1).long()]
        # coeff0 = (gho_cor_a_len[:, 0] * gho_cor_a_len[:, 1]) / ((len0 + len1) * len0)
        # self.NodeGhoCoeff[corner_index[ghost_corner_a_index[:, 0]], ghost_corner_a_index[:, 1], 2] = coeff0
        # coeff1 = (gho_cor_a_len[:, 0] * gho_cor_a_len[:, 1]) / ((len0 + len1) * len1)
        # self.NodeGhoCoeff[corner_index[ghost_corner_a_index[:, 0]], ghost_corner_a_index[:, 1], 3] = coeff1
        # coeff = coeff0 + coeff1
        # self.NodeGhoCoeff[corner_index[ghost_corner_a_index[:, 0]], ghost_corner_a_index[:, 1], 4] = coeff

        ghost_corner_b_mask = (
                self.NodeGhoNeighbors[self.CornerMap[corner_index].long(), :, 3] >= 0)
        ghost_corner_b_index = torch.nonzero(ghost_corner_b_mask, as_tuple=False)

        cor_map_index = self.CornerMap[corner_index[ghost_corner_b_index[:, 0]].long()].long()
        gho_cor_b_len = gho_node_neigh_len[cor_map_index, ghost_corner_b_index[:, 1]]
        gho_common_len = (gho_cor_b_len[:, 0] + gho_cor_b_len[:, 1]) * (gho_cor_b_len[:, 2] + gho_cor_b_len[:, 3])

        coeff = (gho_cor_b_len[:, 1] * gho_cor_b_len[:, 3]) / gho_common_len
        self.NodeGhoCoeff[cor_map_index, ghost_corner_b_index[:, 1], 0] = coeff.to(self.CornerSH.device)
        coeff = (gho_cor_b_len[:, 0] * gho_cor_b_len[:, 3]) / gho_common_len
        self.NodeGhoCoeff[cor_map_index, ghost_corner_b_index[:, 1], 1] = coeff.to(self.CornerSH.device)
        coeff = (gho_cor_b_len[:, 1] * gho_cor_b_len[:, 2]) / gho_common_len
        self.NodeGhoCoeff[cor_map_index, ghost_corner_b_index[:, 1], 2] = coeff.to(self.CornerSH.device)
        coeff = (gho_cor_b_len[:, 0] * gho_cor_b_len[:, 2]) / gho_common_len
        self.NodeGhoCoeff[cor_map_index, ghost_corner_b_index[:, 1], 3] = coeff.to(self.CornerSH.device)
        # gho_node_neigh_len[cor_map_index, ghost_corner_b_index[:, 1], 0] = (gho_node_neigh_len[
        #                                                                         cor_map_index, ghost_corner_b_index[:,
        #                                                                                        1], 1] +
        #                                                                     gho_node_neigh_len[
        #                                                                         cor_map_index, ghost_corner_b_index[:,
        #                                                                                        1], 3]) / gho_common_len
        # gho_node_neigh_len[cor_map_index, ghost_corner_b_index[:, 1], 1] = (gho_node_neigh_len[
        #                                                                         cor_map_index, ghost_corner_b_index[:,
        #                                                                                        1], 0] +
        #                                                                     gho_node_neigh_len[
        #                                                                         cor_map_index, ghost_corner_b_index[:,
        #                                                                                        1], 3]) / gho_common_len
        # gho_node_neigh_len[cor_map_index, ghost_corner_b_index[:, 1], 2] = (gho_node_neigh_len[
        #                                                                         cor_map_index, ghost_corner_b_index[:,
        #                                                                                        1], 1] +
        #                                                                     gho_node_neigh_len[
        #                                                                         cor_map_index, ghost_corner_b_index[:,
        #                                                                                        1], 2]) / gho_common_len
        # gho_node_neigh_len[cor_map_index, ghost_corner_b_index[:, 1], 3] = (gho_node_neigh_len[
        #                                                                         cor_map_index, ghost_corner_b_index[:,
        #                                                                                        1], 0] +
        #                                                                     gho_node_neigh_len[
        #                                                                         cor_map_index, ghost_corner_b_index[:,
        #                                                                                        1], 2]) / gho_common_len

        # self.NodeGhoCoeff = gho_node_neigh_len

        # pos_dir_index = torch.zeros(ghost_corner_b_index.size(0), 2, dtype=torch.int32, device=self.CornerSH.device)
        # gcb_index = (ghost_corner_b_index[:, 1] == 0).nonzero()
        # pos_dir_index[gcb_index, 0] = 1
        # pos_dir_index[gcb_index, 1] = 2
        # gcb_index = (ghost_corner_b_index[:, 1] == 1).nonzero()
        # pos_dir_index[gcb_index, 0] = 0
        # pos_dir_index[gcb_index, 1] = 2
        # gcb_index = (ghost_corner_b_index[:, 1] == 2).nonzero()
        # pos_dir_index[gcb_index, 0] = 1
        # pos_dir_index[gcb_index, 1] = 0
        # pos_dir_index = pos_dir_index.long()
        #
        # len0 = self.NodeAllNeighLen[corner_index[ghost_corner_b_index[:, 0]], (pos_dir_index[:, 0] * 2).long()]
        # len1 = self.NodeAllNeighLen[corner_index[ghost_corner_b_index[:, 0]], (pos_dir_index[:, 0] * 2 + 1).long()]
        # coeff0 = (gho_cor_b_len[:, 0] * gho_cor_b_len[:, 1]) / ((len0 + len1) * len0)
        # self.NodeGhoCoeff[corner_index[ghost_corner_b_index[:, 0]], ghost_corner_b_index[:, 1], 4] = coeff0
        # coeff1 = (gho_cor_b_len[:, 0] * gho_cor_b_len[:, 1]) / ((len0 + len1) * len1)
        # self.NodeGhoCoeff[corner_index[ghost_corner_b_index[:, 0]], ghost_corner_b_index[:, 1], 5] = coeff1
        #
        # len0 = self.NodeAllNeighLen[corner_index[ghost_corner_b_index[:, 0]], (pos_dir_index[:, 1] * 2).long()]
        # len1 = self.NodeAllNeighLen[corner_index[ghost_corner_b_index[:, 0]], (pos_dir_index[:, 1] * 2 + 1).long()]
        # coeff2 = (gho_cor_b_len[:, 2] * gho_cor_b_len[:, 3]) / ((len0 + len1) * len0)
        # self.NodeGhoCoeff[corner_index[ghost_corner_b_index[:, 0]], ghost_corner_b_index[:, 1], 6] = coeff2
        # coeff3 = (gho_cor_b_len[:, 2] * gho_cor_b_len[:, 3]) / ((len0 + len1) * len1)
        # self.NodeGhoCoeff[corner_index[ghost_corner_b_index[:, 0]], ghost_corner_b_index[:, 1], 7] = coeff3
        # coeff = coeff0 + coeff1 + coeff2 + coeff3
        # self.NodeGhoCoeff[corner_index[ghost_corner_b_index[:, 0]], ghost_corner_b_index[:, 1], 8] = coeff

    def FindGhostNeighbors(self):
        corner_node_list = torch.zeros(self.CornerSH.size(0), 4, dtype=torch.int32,
                                       device=self.CornerSH.device)

        corner_node_list_2_0 = -1 * torch.ones(self.CornerSH.size(0), 5, 1, dtype=torch.int32,
                                               device=self.CornerSH.device)
        corner_node_list_2_1 = -1 * torch.ones(self.CornerSH.size(0), 5, 3, dtype=torch.int8,
                                               device=self.CornerSH.device)

        corner_node_depth = 999 * torch.ones(self.CornerSH.size(0), 1, dtype=torch.int32,
                                             device=self.CornerSH.device)

        sel = (*self._all_leaves().T,)
        depths = self.parent_depth[sel[0], 1]

        leaf_node = torch.stack(sel, dim=-1).to(device=self.data.device)
        for i in range(8):
            leaf_depth = corner_node_depth[self.NodeCorners[leaf_node[:, 0].long(),
            leaf_node[:, 1].long(),
            leaf_node[:, 2].long(),
            leaf_node[:, 3].long()][:, i].long()][:, 0]
            depth_mask = (leaf_depth > depths)

            if depth_mask.nonzero().numel() > 0:
                c_leaf_node = leaf_node[depth_mask].type(torch.int32)
                corner_node_list[self.NodeCorners[c_leaf_node[:, 0].long(),
                c_leaf_node[:, 1].long(),
                c_leaf_node[:, 2].long(),
                c_leaf_node[:, 3].long()][:, i].long()] = c_leaf_node[:]
                c_leaf_depth = depths[depth_mask]
                corner_node_depth[self.NodeCorners[c_leaf_node[:, 0].long(),
                c_leaf_node[:, 1].long(),
                c_leaf_node[:, 2].long(),
                c_leaf_node[:, 3].long()][:, i].long(), 0] = c_leaf_depth

            if i < 5:
                corner_node_list_2_0[self.NodeCorners[leaf_node[:, 0].long(),
                leaf_node[:, 1].long(),
                leaf_node[:, 2].long(),
                leaf_node[:, 3].long()][:, i].long(), i, 0] = leaf_node[:, 0].type(torch.int32)

                corner_node_list_2_1[self.NodeCorners[leaf_node[:, 0].long(),
                leaf_node[:, 1].long(),
                leaf_node[:, 2].long(),
                leaf_node[:, 3].long()][:, i].long(), i, 0:3] = leaf_node[:, 1:4].type(torch.int8)

        reso = 2 ** (self.depth_limit + 1) + 1

        indices_x_valid = self.ValidGeoCornerCoord[:, 0].long()
        indices_y_valid = self.ValidGeoCornerCoord[:, 1].long()
        indices_z_valid = self.ValidGeoCornerCoord[:, 2].long()
        corner_index = self.ValidGeoCorner.long()

        indices_x_range_mask = (indices_x_valid - 1) >= 0
        indices_x_valid = indices_x_valid[indices_x_range_mask]
        indices_y_valid = indices_y_valid[indices_x_range_mask]
        indices_z_valid = indices_z_valid[indices_x_range_mask]
        corner_index = corner_index[indices_x_range_mask]
        indices_x_range_mask = (indices_x_valid + 1) < reso
        indices_x_valid = indices_x_valid[indices_x_range_mask]
        indices_y_valid = indices_y_valid[indices_x_range_mask]
        indices_z_valid = indices_z_valid[indices_x_range_mask]
        corner_index = corner_index[indices_x_range_mask]

        indices_y_range_mask = (indices_y_valid - 1) >= 0
        indices_x_valid = indices_x_valid[indices_y_range_mask]
        indices_y_valid = indices_y_valid[indices_y_range_mask]
        indices_z_valid = indices_z_valid[indices_y_range_mask]
        corner_index = corner_index[indices_y_range_mask]
        indices_y_range_mask = (indices_y_valid + 1) < reso
        indices_x_valid = indices_x_valid[indices_y_range_mask]
        indices_y_valid = indices_y_valid[indices_y_range_mask]
        indices_z_valid = indices_z_valid[indices_y_range_mask]
        corner_index = corner_index[indices_y_range_mask]

        indices_z_range_mask = (indices_z_valid - 1) >= 0
        indices_x_valid = indices_x_valid[indices_z_range_mask]
        indices_y_valid = indices_y_valid[indices_z_range_mask]
        indices_z_valid = indices_z_valid[indices_z_range_mask]
        corner_index = corner_index[indices_z_range_mask]
        indices_z_range_mask = (indices_z_valid + 1) < reso
        indices_x_valid = indices_x_valid[indices_z_range_mask]
        indices_y_valid = indices_y_valid[indices_z_range_mask]
        indices_z_valid = indices_z_valid[indices_z_range_mask]
        corner_index = corner_index[indices_z_range_mask]

        indices_x_valid = indices_x_valid.unsqueeze(0).T
        indices_y_valid = indices_y_valid.unsqueeze(0).T
        indices_z_valid = indices_z_valid.unsqueeze(0).T

        corner_pos = torch.cat([indices_x_valid, indices_y_valid, indices_z_valid], dim=1)
        corner_pos = corner_pos.type(torch.float32) / (reso - 1)

        gho_node_neigh_len = -1 * torch.ones(self.ValidGeoCorner.size(0), 3, 4, dtype=torch.float32, device='cpu')

        wo_c_all_corner = []
        for i in range(6):
            wo_ci_corner_mask = (self.NodeAllNeighbors[self.CornerMap[corner_index].long()][:, i] == -1)
            wo_ci_corner = corner_index[wo_ci_corner_mask]
            wo_ci_corner_pos = corner_pos[wo_ci_corner_mask]
            self.FindGhoNeighCore(corner_node_list, corner_node_list_2_0, corner_node_list_2_1, wo_ci_corner,
                                  wo_ci_corner_pos, gho_node_neigh_len, i)

            wo_c_all_corner.append(wo_ci_corner)

        wo_c_all_corner = torch.cat(wo_c_all_corner, dim=0).unique()
        self.CalGhostCoeff(wo_c_all_corner, gho_node_neigh_len)

    def FindGaussianNeigh(self, ksize, sigma):
        node_gauss_neigh = -1 * torch.ones(self.ValidGeoCorner.size(0), ksize, ksize, ksize, dtype=torch.int32,
                                           device=self.CornerSDF.device)

        reso_indices_x = self.ValidGeoCornerCoord[:, 0]
        reso_indices_y = self.ValidGeoCornerCoord[:, 1]
        reso_indices_z = self.ValidGeoCornerCoord[:, 2]
        corner_indices = self.ValidGeoCorner.long()

        current_reso = 2 ** (self._c_depth + 1)
        max_reso = 2 ** (self.depth_limit + 1) + 1
        reso_offset = (max_reso - 1) / current_reso

        for i in range(int(-ksize / 2), int(ksize / 2) + 1):
            for j in range(int(-ksize / 2), int(ksize / 2) + 1):
                for k in range(int(-ksize / 2), int(ksize / 2) + 1):
                    c_x_indices = reso_indices_x + i * reso_offset
                    c_y_indices = reso_indices_y + j * reso_offset
                    c_z_indices = reso_indices_z + k * reso_offset

                    c_xyz_valid_mask = (c_x_indices < 0)
                    c_valid_mask = c_xyz_valid_mask
                    c_xyz_valid_mask = (c_y_indices < 0)
                    c_valid_mask = c_valid_mask + c_xyz_valid_mask
                    c_xyz_valid_mask = (c_z_indices < 0)
                    c_valid_mask = c_valid_mask + c_xyz_valid_mask
                    c_xyz_valid_mask = (c_x_indices > max_reso - 1)
                    c_valid_mask = c_valid_mask + c_xyz_valid_mask
                    c_xyz_valid_mask = (c_y_indices > max_reso - 1)
                    c_valid_mask = c_valid_mask + c_xyz_valid_mask
                    c_xyz_valid_mask = (c_z_indices > max_reso - 1)
                    c_valid_mask = c_valid_mask + c_xyz_valid_mask
                    c_valid_mask = ~c_valid_mask

                    c_valid_indices = corner_indices[c_valid_mask]
                    c_x_indices = c_x_indices[c_valid_mask]
                    c_y_indices = c_y_indices[c_valid_mask]
                    c_z_indices = c_z_indices[c_valid_mask]

                    c_xyz_indices = torch.stack((c_x_indices, c_y_indices, c_z_indices), dim=1)

                    c_corner_indices = self.FromPosGetCornerIndex(c_xyz_indices)

                    geo_corner_mask = (self.CornerMap[c_corner_indices.long()] == -1)
                    c_corner_indices[geo_corner_mask] = -1

                    node_gauss_neigh[self.CornerMap[c_valid_indices].long(),
                    i + int(ksize / 2), j + int(ksize / 2), k + int(ksize / 2)] = c_corner_indices

        xx, yy, zz = torch.meshgrid(torch.arange(-(ksize // 2), ksize // 2 + 1, 1),
                                    torch.arange(-(ksize // 2), ksize // 2 + 1, 1),
                                    torch.arange(-(ksize // 2), ksize // 2 + 1, 1))

        node_gauss_kernal = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
        node_gauss_kernal = torch.unsqueeze(node_gauss_kernal, dim=0)
        node_gauss_kernal = node_gauss_kernal.repeat(self.ValidGeoCorner.size(0), 1, 1, 1)

        valid_corner_mask = (node_gauss_neigh < 0)
        node_gauss_kernal[valid_corner_mask] = 0.0
        node_gauss_kernal = node_gauss_kernal.view(self.ValidGeoCorner.size(0), -1)
        kernal_sum = torch.sum(node_gauss_kernal, dim=1)
        kernal_sum = kernal_sum.unsqueeze(dim=1)
        kernal_sum = kernal_sum.repeat(1, ksize * ksize * ksize)
        node_gauss_kernal /= kernal_sum
        node_gauss_kernal = node_gauss_kernal.view(self.ValidGeoCorner.size(0), ksize, ksize, ksize)
        node_gauss_kernal = node_gauss_kernal.to(self.CornerSDF.device)

        node_gauss_neigh = node_gauss_neigh.view(-1, ksize * ksize * ksize)
        node_gauss_kernal = node_gauss_kernal.view(-1, ksize * ksize * ksize)

        return node_gauss_neigh, node_gauss_kernal

    def FindCubicGaussianNeigh(self, sigma):
        ksize = 3

        sel = (*self._all_leaves().T,)
        leaf_node = torch.stack(sel, dim=-1).to(device=self.CornerSH.device)
        leaf_depth = self.parent_depth[sel[0], 1]

        leaf_mask = (leaf_depth == self._c_depth)
        leaf_node = leaf_node[leaf_mask]

        node_gauss_neigh = -1 * torch.ones(self.CornerSH.size(0), ksize, ksize, ksize, dtype=torch.int32,
                                           device=self.CornerSDF.device)

        leaf_node = leaf_node.long()
        for i in range(8):
            c_node_corner = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, i]
            if i == 0:
                node_gauss_neigh[c_node_corner[:].long(), 1, 1, 1] = c_node_corner

                c_node_corner = c_node_corner.long()

                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 1]
                node_gauss_neigh[c_node_corner[:], 1, 2, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 2]
                node_gauss_neigh[c_node_corner[:], 2, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 3]
                node_gauss_neigh[c_node_corner[:], 2, 2, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 4]
                node_gauss_neigh[c_node_corner[:], 1, 1, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 5]
                node_gauss_neigh[c_node_corner[:], 1, 2, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 6]
                node_gauss_neigh[c_node_corner[:], 2, 1, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 7]
                node_gauss_neigh[c_node_corner[:], 2, 2, 2] = c_node_nei_corner
            elif i == 1:
                c_node_corner = c_node_corner.long()

                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 0]
                node_gauss_neigh[c_node_corner[:], 1, 0, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 2]
                node_gauss_neigh[c_node_corner[:], 2, 0, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 4]
                node_gauss_neigh[c_node_corner[:], 1, 0, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 6]
                node_gauss_neigh[c_node_corner[:], 2, 0, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 1]
                node_gauss_neigh[c_node_corner[:], 1, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 3]
                node_gauss_neigh[c_node_corner[:], 2, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 5]
                node_gauss_neigh[c_node_corner[:], 1, 1, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 7]
                node_gauss_neigh[c_node_corner[:], 2, 1, 2] = c_node_nei_corner
            elif i == 2:
                c_node_corner = c_node_corner.long()

                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 0]
                node_gauss_neigh[c_node_corner[:], 0, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 1]
                node_gauss_neigh[c_node_corner[:], 0, 2, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 4]
                node_gauss_neigh[c_node_corner[:], 0, 1, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 5]
                node_gauss_neigh[c_node_corner[:], 0, 2, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 2]
                node_gauss_neigh[c_node_corner[:], 1, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 3]
                node_gauss_neigh[c_node_corner[:], 1, 2, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 6]
                node_gauss_neigh[c_node_corner[:], 1, 1, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 7]
                node_gauss_neigh[c_node_corner[:], 1, 2, 2] = c_node_nei_corner
            elif i == 3:
                c_node_corner = c_node_corner.long()

                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 0]
                node_gauss_neigh[c_node_corner[:], 0, 0, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 4]
                node_gauss_neigh[c_node_corner[:], 0, 0, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 1]
                node_gauss_neigh[c_node_corner[:], 0, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 5]
                node_gauss_neigh[c_node_corner[:], 0, 1, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 2]
                node_gauss_neigh[c_node_corner[:], 1, 0, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 6]
                node_gauss_neigh[c_node_corner[:], 1, 0, 2] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 3]
                node_gauss_neigh[c_node_corner[:], 1, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 7]
                node_gauss_neigh[c_node_corner[:], 1, 1, 2] = c_node_nei_corner
            elif i == 4:
                c_node_corner = c_node_corner.long()

                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 0]
                node_gauss_neigh[c_node_corner[:], 1, 1, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 1]
                node_gauss_neigh[c_node_corner[:], 1, 2, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 2]
                node_gauss_neigh[c_node_corner[:], 2, 1, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 3]
                node_gauss_neigh[c_node_corner[:], 2, 2, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 4]
                node_gauss_neigh[c_node_corner[:], 1, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 5]
                node_gauss_neigh[c_node_corner[:], 1, 2, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 6]
                node_gauss_neigh[c_node_corner[:], 2, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 7]
                node_gauss_neigh[c_node_corner[:], 2, 2, 1] = c_node_nei_corner
            elif i == 5:
                c_node_corner = c_node_corner.long()

                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 0]
                node_gauss_neigh[c_node_corner[:], 1, 0, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 2]
                node_gauss_neigh[c_node_corner[:], 2, 0, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 4]
                node_gauss_neigh[c_node_corner[:], 1, 0, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 6]
                node_gauss_neigh[c_node_corner[:], 2, 0, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 1]
                node_gauss_neigh[c_node_corner[:], 1, 1, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 3]
                node_gauss_neigh[c_node_corner[:], 2, 1, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 5]
                node_gauss_neigh[c_node_corner[:], 1, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 7]
                node_gauss_neigh[c_node_corner[:], 2, 1, 1] = c_node_nei_corner
            elif i == 6:
                c_node_corner = c_node_corner.long()

                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 0]
                node_gauss_neigh[c_node_corner[:], 0, 1, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 1]
                node_gauss_neigh[c_node_corner[:], 0, 2, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 2]
                node_gauss_neigh[c_node_corner[:], 1, 1, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 3]
                node_gauss_neigh[c_node_corner[:], 1, 2, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 4]
                node_gauss_neigh[c_node_corner[:], 0, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 5]
                node_gauss_neigh[c_node_corner[:], 0, 2, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 6]
                node_gauss_neigh[c_node_corner[:], 1, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 7]
                node_gauss_neigh[c_node_corner[:], 1, 2, 1] = c_node_nei_corner
            elif i == 7:
                c_node_corner = c_node_corner.long()

                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 0]
                node_gauss_neigh[c_node_corner[:], 0, 0, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 1]
                node_gauss_neigh[c_node_corner[:], 0, 1, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 2]
                node_gauss_neigh[c_node_corner[:], 1, 0, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 3]
                node_gauss_neigh[c_node_corner[:], 1, 1, 0] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 4]
                node_gauss_neigh[c_node_corner[:], 0, 0, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 5]
                node_gauss_neigh[c_node_corner[:], 0, 1, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 6]
                node_gauss_neigh[c_node_corner[:], 1, 0, 1] = c_node_nei_corner
                ##################################################################################################
                c_node_nei_corner = self.NodeCorners[
                                        leaf_node[:, 0], leaf_node[:, 1], leaf_node[:, 2], leaf_node[:, 3]][:, 7]
                node_gauss_neigh[c_node_corner[:], 1, 1, 1] = c_node_nei_corner

        node_gauss_neigh = node_gauss_neigh[self.ValidGeoCorner.long()]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    c_gauss_corner = node_gauss_neigh[:, i, j, k]
                    gauss_corner_mask = (self.CornerMap[c_gauss_corner.long()] == -1)
                    c_gauss_corner[gauss_corner_mask] = -1
                    node_gauss_neigh[:, i, j, k] = c_gauss_corner

        xx, yy, zz = torch.meshgrid(torch.arange(-(ksize // 2), ksize // 2 + 1, 1),
                                    torch.arange(-(ksize // 2), ksize // 2 + 1, 1),
                                    torch.arange(-(ksize // 2), ksize // 2 + 1, 1))

        node_gauss_kernal = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
        node_gauss_kernal = torch.unsqueeze(node_gauss_kernal, dim=0)
        node_gauss_kernal = node_gauss_kernal.repeat(self.ValidGeoCorner.size(0), 1, 1, 1)

        chunk_size = int(1e5)
        for i in tqdm(range(0, node_gauss_kernal.size(0), chunk_size)):
            valid_corner_mask = (node_gauss_neigh[i:i + chunk_size] < 0)
            c_gauss_kernal = node_gauss_kernal[i:i + chunk_size]
            c_gauss_kernal[valid_corner_mask] = 0.0
            node_gauss_kernal[i:i + chunk_size] = c_gauss_kernal

        node_gauss_kernal = node_gauss_kernal.view(self.ValidGeoCorner.size(0), -1)
        kernal_sum = torch.sum(node_gauss_kernal, dim=1)
        kernal_sum = kernal_sum.unsqueeze(dim=1)
        kernal_sum = kernal_sum.repeat(1, ksize * ksize * ksize)
        node_gauss_kernal /= kernal_sum
        node_gauss_kernal = node_gauss_kernal.view(self.ValidGeoCorner.size(0), ksize, ksize, ksize)
        node_gauss_kernal = node_gauss_kernal.to(self.CornerSDF.device)

        node_gauss_neigh = node_gauss_neigh.view(-1, ksize * ksize * ksize)
        node_gauss_kernal = node_gauss_kernal.view(-1, ksize * ksize * ksize)

        return node_gauss_neigh, node_gauss_kernal

    def VolumeRender(
            self,
            rays: Rays,
            rgb_gt: torch.Tensor,
            beta_loss: float = 0.0,
            sparsity_loss: float = 0.0
    ):
        grad_density, grad_sh = self.GetDataGrads()
        rgb_out = torch.zeros_like(rgb_gt)

        grad_holder = _C.GridOutputGradsLOT()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh

        cu_fn = _C.__dict__[f"volume_render_{self.opt.backend}_fused_LOT"]
        #  with utils.Timing("actual_render"):
        cu_fn(
            self._LOTspecA(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            rgb_gt,
            beta_loss,
            sparsity_loss,
            rgb_out,
            grad_holder
        )

        # result1 = self.CornerD.grad.cpu().numpy()
        # np.savetxt('D:/Lotree/npresult5.txt', result1)

        return rgb_out

    def VolumeRenderDS(
            self,
            rays: Rays,
            rgb_gt: torch.Tensor
    ):
        grad_sdf, grad_sh, grad_beta, grad_s = self.GetDataGradsVolSDF()
        rgb_out = torch.zeros_like(rgb_gt)

        grad_holder = _C.GridOutputGradsSDFLOT()
        grad_holder.grad_sdf_out = grad_sdf
        grad_holder.grad_sh_out = grad_sh
        grad_holder.grad_beta_out = grad_beta
        grad_holder.grad_learns_out = grad_s

        cu_fn = _C.__dict__[f"volume_render_volsdf_downsample_fused_LOT"]

        cu_fn(
            self._LOTspecA(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            rgb_gt,
            rgb_out,
            self.LeafNodeMap,
            grad_holder
        )

        return rgb_out

    def VolumeRenderConvert(
            self,
            rays: Rays,
            rgb_gt: torch.Tensor,
            scale: float = 2e-4,
            ksize_grad: int = 3,
            sparse_frac: float = 0.1,
            gauss_grad_loss=False
    ):
        grad_sdf, grad_sh, grad_beta, grad_s = self.GetDataGradsVolSDF()
        rgb_out = torch.zeros_like(rgb_gt)

        grad_holder = _C.GridOutputGradsSDFLOT()
        grad_holder.grad_sdf_out = grad_sdf
        grad_holder.grad_sh_out = grad_sh
        grad_holder.grad_beta_out = grad_beta
        grad_holder.grad_learns_out = grad_s

        cu_fn = _C.__dict__[f"volume_render_volsdf_convert_fused_LOT"]

        cuda_tree = self._LOTspecA()
        cuda_opt = self.opt._to_cpp()

        cu_fn(
            cuda_tree,
            rays._to_cpp(),
            cuda_opt,
            rgb_gt,
            rgb_out,
            grad_holder
        )

        if gauss_grad_loss:
            smooth_loss = torch.zeros(self.ValidGeoCorner.size(0), 1, dtype=torch.float32, device=
            self.CornerSDF.device)

            if sparse_frac != 1.0:
                rand_cells = self.GetRandCells(sparse_frac, contiguous=True)
            else:
                rand_cells = self.ValidGeoCorner

            corner_gradient = torch.zeros(self.ValidGeoCorner.size(0), 3, dtype=torch.float32, device=
            self.CornerSDF.device)

            _C.com_corner_gradient_LOT(self.NodeAllNeighbors, self.NodeAllNeighLen,
                                       self.NodeGhoNeighbors, self.NodeGhoCoeff,
                                       self.CornerMap,
                                       self.CornerSDF,
                                       rand_cells,
                                       corner_gradient)

            gauss_gradient_diff = torch.zeros(self.ValidGeoCorner.size(0), 3, dtype=torch.float32, device=
            self.CornerSDF.device)

            _C.gaussian_gradient_conv_LOT(self.NodeGaussGradNeighbors, self.NodeGaussGradKernals,
                                          self.CornerMap,
                                          corner_gradient,
                                          rand_cells,
                                          0, ksize_grad * ksize_grad * ksize_grad,
                                          gauss_gradient_diff)

            _C.gauss_gradient_smooth_fused_LOT(self.NodeAllNeighbors, self.NodeGhoNeighbors,
                                               self.NodeGaussGradNeighbors,
                                               self.NodeAllNeighLen, self.NodeGhoCoeff, self.NodeGaussGradKernals,
                                               self.CornerMap,
                                               gauss_gradient_diff,
                                               rand_cells,
                                               scale,
                                               0, ksize_grad * ksize_grad * ksize_grad,
                                               smooth_loss,
                                               self.CornerSDF.grad)

            smooth_loss = smooth_loss.sum(0)
            smooth_loss[0] = smooth_loss[0] / rand_cells.size(0)

        return rgb_out

    def VolumeRenderGaussVolSDF(
            self,
            rays: Rays,
            rgb_gt: torch.Tensor,
            scale: float = 2e-4,
            ksize: int = 5,
            ksize_grad: int = 3,
            sparse_frac: float = 0.1
    ):
        """
        c_data = self.CornerSDF[self.NodeGaussNeighbors[:].long()].squeeze()
        self.CornerGaussSDF.data = self.NodeGaussKernals * c_data
        self.CornerGaussSDF.data = self.CornerGaussSDF.sum(1).unsqueeze(1)
        """

        # comp_cells = torch.arange(0, self.CornerSDF.size(0), dtype=torch.int32, device=
        # self.CornerSDF.device)

        self.CornerGaussSDF.data = torch.zeros(self.CornerSDF.size(0), 1, dtype=torch.float32, device=
        self.CornerSDF.device)

        _C.gaussian_sdf_conv_LOT(self.NodeGaussNeighbors,
                                 self.NodeGaussKernals,
                                 self.CornerMap,
                                 self.CornerSDF,
                                 self.ValidGeoCorner,
                                 0, ksize * ksize * ksize,
                                 self.CornerGaussSDF)

        grad_sdf, grad_sh, grad_beta, grad_s = self.GetDataGradsVolSDF()
        rgb_out = torch.zeros_like(rgb_gt)

        grad_holder = _C.GridOutputGradsSDFLOT()
        grad_holder.grad_sdf_out = grad_sdf
        grad_holder.grad_sh_out = grad_sh
        grad_holder.grad_beta_out = grad_beta
        grad_holder.grad_learns_out = grad_s

        cuda_tree = self._LOTspecA()
        cuda_opt = self.opt._to_cpp()

        cu_fn = _C.__dict__[f"volume_render_volsdf_gaussian_fused_LOT"]
        #  with utils.Timing("actual_render"):

        cu_fn(
            cuda_tree,
            rays._to_cpp(),
            cuda_opt,
            rgb_gt,
            rgb_out,
            grad_holder
        )

        gauss_sdf_grad = torch.zeros(self.ValidGeoCorner.size(0), 1, dtype=torch.float32, device=
        self.CornerSDF.device)

        _C.gaussian_sdf_conv_backward_LOT(self.NodeGaussNeighbors,
                                          self.NodeGaussKernals,
                                          self.CornerMap,
                                          self.CornerSDF.grad,
                                          self.ValidGeoCorner,
                                          0, ksize * ksize * ksize,
                                          gauss_sdf_grad)

        self.CornerSDF.grad.data = gauss_sdf_grad.clone()

        if sparse_frac != 1.0:
            rand_cells = self.GetRandCells(sparse_frac, contiguous=True)
        else:
            rand_cells = self.ValidGeoCorner

        smooth_loss = torch.zeros(self.ValidGeoCorner.size(0), 1, dtype=torch.float32, device=
        self.CornerSDF.device)

        corner_gradient = torch.zeros(self.ValidGeoCorner.size(0), 3, dtype=torch.float32, device=
        self.CornerSDF.device)

        _C.com_corner_gradient_LOT(self.NodeAllNeighbors, self.NodeAllNeighLen,
                                   self.NodeGhoNeighbors, self.NodeGhoCoeff,
                                   self.CornerMap,
                                   self.CornerSDF,
                                   rand_cells,
                                   corner_gradient)

        gauss_gradient_diff = torch.zeros(self.ValidGeoCorner.size(0), 3, dtype=torch.float32, device=
        self.CornerSDF.device)

        _C.gaussian_gradient_conv_LOT(self.NodeGaussGradNeighbors, self.NodeGaussGradKernals,
                                      self.CornerMap,
                                      corner_gradient,
                                      rand_cells,
                                      0, ksize_grad * ksize_grad * ksize_grad,
                                      gauss_gradient_diff)

        _C.gauss_gradient_smooth_fused_LOT(self.NodeAllNeighbors, self.NodeGhoNeighbors,
                                           self.NodeGaussGradNeighbors,
                                           self.NodeAllNeighLen, self.NodeGhoCoeff, self.NodeGaussGradKernals,
                                           self.CornerMap,
                                           gauss_gradient_diff,
                                           rand_cells,
                                           scale,
                                           0, ksize_grad * ksize_grad * ksize_grad,
                                           smooth_loss,
                                           self.CornerSDF.grad)

        smooth_loss = smooth_loss.sum(0)
        smooth_loss[0] = smooth_loss[0] / rand_cells.size(0)

        return rgb_out, smooth_loss[0]

    def VolumeRenderWOGaussVolSDF(
            self,
            rays: Rays,
            rgb_gt: torch.Tensor,
            scale: float = 2e-4,
            ksize_grad: int = 3,
            sparse_frac: float = 0.1,
            gauss_grad_loss=False
    ):
        """
        c_data = self.CornerSDF[self.NodeGaussNeighbors[:].long()].squeeze()
        self.CornerGaussSDF.data = self.NodeGaussKernals * c_data
        self.CornerGaussSDF.data = self.CornerGaussSDF.sum(1).unsqueeze(1)
        """

        # comp_cells = torch.arange(0, self.CornerSDF.size(0), dtype=torch.int32, device=
        # self.CornerSDF.device)

        grad_sdf, grad_sh, grad_beta, grad_s = self.GetDataGradsVolSDF()
        rgb_out = torch.zeros_like(rgb_gt)

        grad_holder = _C.GridOutputGradsSDFLOT()
        grad_holder.grad_sdf_out = grad_sdf
        grad_holder.grad_sh_out = grad_sh
        grad_holder.grad_beta_out = grad_beta
        grad_holder.grad_learns_out = grad_s

        # self.CornerGaussSDF.data = self.CornerSDF.data.clone()

        cu_fn = _C.__dict__[f"volume_render_volsdf_fused_LOT"]
        #  with utils.Timing("actual_render"):

        cu_fn(
            self._LOTspecA(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            rgb_gt,
            rgb_out,
            grad_holder
        )

        smooth_loss = torch.zeros(self.ValidGeoCorner.size(0), 1, dtype=torch.float32, device=
        self.CornerSDF.device)

        if gauss_grad_loss:
            if sparse_frac != 1.0:
                rand_cells = self.GetRandCells(sparse_frac, contiguous=True)
            else:
                rand_cells = self.ValidGeoCorner

            corner_gradient = torch.zeros(self.ValidGeoCorner.size(0), 3, dtype=torch.float32, device=
            self.CornerSDF.device)

            _C.com_corner_gradient_LOT(self.NodeAllNeighbors, self.NodeAllNeighLen,
                                       self.NodeGhoNeighbors, self.NodeGhoCoeff,
                                       self.CornerMap,
                                       self.CornerSDF,
                                       rand_cells,
                                       corner_gradient)

            gauss_gradient_diff = torch.zeros(self.ValidGeoCorner.size(0), 3, dtype=torch.float32, device=
            self.CornerSDF.device)

            _C.gaussian_gradient_conv_LOT(self.NodeGaussGradNeighbors, self.NodeGaussGradKernals,
                                          self.CornerMap,
                                          corner_gradient,
                                          rand_cells,
                                          0, ksize_grad * ksize_grad * ksize_grad,
                                          gauss_gradient_diff)

            _C.gauss_gradient_smooth_fused_LOT(self.NodeAllNeighbors, self.NodeGhoNeighbors,
                                               self.NodeGaussGradNeighbors,
                                               self.NodeAllNeighLen, self.NodeGhoCoeff, self.NodeGaussGradKernals,
                                               self.CornerMap,
                                               gauss_gradient_diff,
                                               rand_cells,
                                               scale,
                                               0, ksize_grad * ksize_grad * ksize_grad,
                                               smooth_loss,
                                               self.CornerSDF.grad)

            smooth_loss = smooth_loss.sum(0)
            smooth_loss[0] = smooth_loss[0] / rand_cells.size(0)
        else:
            smooth_loss[0] = 0.0

        return rgb_out, smooth_loss[0]

    def VolumeRenderVolSDFTest(
            self,
            rays: Rays
    ):
        rgb_out = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32, device=self.CornerSH.device)

        cuda_tree = self._LOTspecA()
        cuda_ray = rays._to_cpp()
        cuda_opt = self.opt._to_cpp()

        cu_fn = _C.__dict__[f"volume_render_volsdf_test_LOT"]
        #  with utils.Timing("actual_render"):
        cu_fn(
            cuda_tree,
            cuda_ray,
            cuda_opt,
            rgb_out
        )

        return rgb_out

    def VolumeRenderVolSDFCVTTest(
            self,
            rays: Rays,
    ):
        rgb_out = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32, device=self.CornerSH.device)

        cuda_tree = self._LOTspecA()
        cuda_ray = rays._to_cpp()
        cuda_opt = self.opt._to_cpp()

        cu_fn = _C.__dict__[f"volume_render_volsdf_convert_test_LOT"]
        #  with utils.Timing("actual_render"):
        cu_fn(
            cuda_tree,
            cuda_ray,
            cuda_opt,
            rgb_out
        )

        return rgb_out

    def VolumeRenderVolSDFDSTest(
            self,
            rays: Rays
    ):
        rgb_out = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32, device=self.CornerSH.device)

        cuda_tree = self._LOTspecA()
        cuda_ray = rays._to_cpp()
        cuda_opt = self.opt._to_cpp()

        cu_fn = _C.__dict__[f"volume_render_volsdf_downsample_test_LOT"]
        #  with utils.Timing("actual_render"):
        cu_fn(
            cuda_tree,
            cuda_ray,
            cuda_opt,
            self.LeafNodeMap,
            rgb_out
        )

        return rgb_out

    def VolumeRenderDSRender(
            self,
            rays: Rays
    ):
        rgb_out = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32, device=self.CornerSH.device)

        cuda_tree = self._LOTspecA()
        cuda_ray = rays._to_cpp()
        cuda_opt = self.opt._to_cpp()

        cu_fn = _C.__dict__[f"volume_render_volsdf_downsample_render_LOT"]
        #  with utils.Timing("actual_render"):
        cu_fn(
            cuda_tree,
            cuda_ray,
            cuda_opt,
            rgb_out
        )

        return rgb_out

    def VolumeRenderGaussVolSDFTest(
            self,
            rays: Rays,
            ksize: int = 5
    ):
        comp_cells = torch.arange(0, self.CornerSDF.size(0), dtype=torch.int32, device=
        self.CornerSDF.device)

        rgb_gt = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32, device=self.CornerSDF.device)

        self.CornerGaussSDF.data = torch.zeros(self.CornerSDF.size(0), 1, dtype=torch.float32, device=
        self.CornerSDF.device)

        _C.gaussian_sdf_conv_LOT(self.NodeGaussNeighbors,
                                 self.NodeGaussKernals,
                                 self.CornerSDF,
                                 comp_cells,
                                 0, ksize * ksize * ksize,
                                 self.CornerGaussSDF)

        grad_sdf, grad_sh, grad_beta, grad_s = self.GetDataGradsVolSDF()
        rgb_out = torch.zeros_like(rgb_gt)

        grad_holder = _C.GridOutputGradsSDFLOT()
        grad_holder.grad_sdf_out = grad_sdf
        grad_holder.grad_sh_out = grad_sh
        grad_holder.grad_beta_out = grad_beta
        grad_holder.grad_learns_out = grad_s

        # self.CornerGaussSDF.data = self.CornerSDF.data.clone()

        hit_num = torch.zeros(rays.origins.size(0), 1, dtype=torch.float32, device=self.CornerSH.device)

        cu_fn = _C.__dict__[f"volume_render_volsdf_gaussian_fused_LOT"]
        #  with utils.Timing("actual_render"):

        cu_fn(
            self._LOTspecA(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            rgb_gt,
            rgb_out,
            hit_num,
            grad_holder
        )

        return rgb_out

    def VolumeRenderVolSDFRecord(self,
                                 dset_train):

        chunk_size = 5000

        all_hln = []
        for i in tqdm(range(0, dset_train.rays.origins.size(0), chunk_size)):
            chunk_origins = dset_train.rays.origins[i: i + chunk_size]
            chunk_dirs = dset_train.rays.dirs[i: i + chunk_size]

            rays = Rays(chunk_origins, chunk_dirs)

            hit_stride = self.RayHitNumVOlSDF(rays)

            record_pos_out = 99999 * torch.ones(rays.origins.size(0), hit_stride[0], 9, dtype=torch.float32,
                                                device=self.CornerSH.device)

            presample_t = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32, device=self.CornerSH.device)
            cuda_tree = self._LOTspecA()
            cuda_ray = rays._to_cpp()
            cuda_opt = self.opt._to_cpp()

            # _C.volume_render_volsdf_prehit_LOT(
            #     cuda_tree,
            #     cuda_ray,
            #     cuda_opt,
            #     presample_t
            # )

            _C.volume_render_volsdf_record_w_LOT(cuda_tree,
                                                 cuda_ray,
                                                 cuda_opt,
                                                 record_pos_out)

            record_pos_out_l = record_pos_out.view(-1, 9)
            # record_pos_out_l_valid_mask = (record_pos_out_l[:, 0] != 0.0)
            # record_pos_out_l = record_pos_out_l[record_pos_out_l_valid_mask]

            temp_r = record_pos_out.cpu().numpy()
            # record_pos_out_l = torch.unique(record_pos_out_l, dim=0)

            a = temp_r
            # all_hln.append(record_pos_out_l)

        all_hln_tensor = torch.cat(all_hln, dim=0)
        # all_hln_tensor = torch.unique(all_hln_tensor, dim=0)

        # all_hln_tensor = all_hln_tensor.cpu().numpy()
        # np.savetxt('D:/LOTree/sample_points1.txt', all_hln_tensor)

    def VolumeRenderSDFTest(
            self,
            rays_in: Rays
    ):
        grad_sdf, grad_sh, grad_learns = self.GetDataGradsSDF()

        grad_holder = _C.GridOutputGradsSDFLOT()
        grad_holder.grad_sdf_out = grad_sdf
        grad_holder.grad_learns_out = grad_learns
        grad_holder.grad_sh_out = grad_sh

        reserve_size = 200

        chunk_size = 1000

        all_rgb_out = []
        for i in tqdm(range(0, rays_in.origins.size(0), chunk_size)):
            chunk_origins = rays_in.origins[i: i + chunk_size]
            chunk_dirs = rays_in.dirs[i: i + chunk_size]

            rays = Rays(chunk_origins, chunk_dirs)

            sdf_point = torch.zeros(rays.origins.size(0), reserve_size, 3, dtype=torch.float32,
                                    device=self.CornerSH.device).contiguous()
            col_point = torch.zeros(rays.origins.size(0), reserve_size, 3, dtype=torch.float32,
                                    device=self.CornerSH.device).contiguous()
            hitnode_sdf = -1 * torch.ones(rays.origins.size(0), reserve_size, 8, dtype=torch.int32,
                                          device=self.CornerSH.device).contiguous()
            hitnode_col = -1 * torch.ones(rays.origins.size(0), reserve_size, 8, dtype=torch.int32,
                                          device=self.CornerSH.device).contiguous()
            hitnum = torch.zeros(rays.origins.size(0), 1, dtype=torch.int32,
                                 device=self.CornerSH.device).contiguous()

            lotree_cuda = self._LOTspecA()
            rays_cuda = rays._to_cpp()

            _C.volume_render_hitpoint_sdf_LOT(
                lotree_cuda,
                rays_cuda,
                self.opt.step_size,
                sdf_point,
                col_point,
                hitnode_sdf,
                hitnode_col,
                hitnum)

            hitnum_max = torch.max(hitnum, dim=0)[0]
            sdf_point = sdf_point[:, 0:hitnum_max].contiguous()
            col_point = col_point[:, 0:hitnum_max].contiguous()
            hitnode_sdf = hitnode_sdf[:, 0:hitnum_max].contiguous()
            hitnode_col = hitnode_col[:, 0:hitnum_max].contiguous()

            rays_hls = RaysHitLOTSDF(sdf_point, col_point, hitnode_sdf, hitnode_col, hitnum)

            rays_hls_cuda = rays_hls._to_cpp()

            rgb_out = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32,
                                  device=self.CornerSH.device).contiguous()

            _C.volume_render_sdf_test_LOT(
                lotree_cuda,
                rays_hls_cuda,
                hitnum_max,
                rays_cuda,
                self.opt._to_cpp(),
                rgb_out)

            all_rgb_out.append(rgb_out)

        all_rgb_tensor = torch.cat(all_rgb_out, dim=0)

        return all_rgb_tensor

    def RayHitNum(
            self,
            rays: Rays):

        hitnum = torch.zeros(rays.origins.size(0), 1, dtype=torch.int32, device=self.CornerSH.device)

        _C.volume_render_hitnum_LOT(
            self._LOTspecA(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            hitnum
        )

        return hitnum.max(0)

    def RayHitNumVOlSDF(
            self,
            rays: Rays):

        hitnum = torch.zeros(rays.origins.size(0), 1, dtype=torch.int32, device=self.CornerSH.device)

        _C.volume_render_volsdf_hitnum_LOT(
            self._LOTspecA(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            hitnum
        )

        return hitnum.max(0)

    def AdaptiveResolution(
            self,
            dset_train,
            weight_threshold: float = 0.0,
            loss_threshold: float = 0.0):

        weight_out = torch.zeros(self.child.size(0), self.child.size(1), self.child.size(2), self.child.size(3), 1,
                                 dtype=torch.float32, device=self.CornerSH.device)

        chunk_size = int(dset_train.h * dset_train.w)

        all_hln = []
        for i in tqdm(range(0, dset_train.rays.origins.size(0), chunk_size)):
            chunk_origins = dset_train.rays.origins[i: i + chunk_size]
            chunk_dirs = dset_train.rays.dirs[i: i + chunk_size]

            im_gt = dset_train.rays.gt[i: i + chunk_size]

            rays = Rays(chunk_origins, chunk_dirs)

            hit_stride = self.RayHitNum(rays)

            rgb_out = torch.zeros_like(im_gt)
            hitnode_out = -1 * torch.ones(rays.origins.size(0), hit_stride[0], 4, dtype=torch.int32,
                                          device=self.CornerSH.device)

            _C.volume_render_refine_LOT(self._LOTspecA(),
                                        rays._to_cpp(),
                                        self.opt._to_cpp(),
                                        rgb_out,
                                        hitnode_out,
                                        weight_out)

            mse = ((rgb_out - im_gt) ** 2)
            mse_min = torch.min(mse, 1)[0]

            higher_loss_mask = (mse_min > loss_threshold)
            hitnode_out_hl = hitnode_out[higher_loss_mask]
            hitnode_out_hl = hitnode_out_hl.view(-1, 4)
            hitnode_out_valid_mask = (hitnode_out_hl[:, 0] != -1)
            hitnode_out_hl = hitnode_out_hl[hitnode_out_valid_mask]
            hitnode_out_hl = torch.unique(hitnode_out_hl, dim=0)

            all_hln.append(hitnode_out_hl)

        all_hln_tensor = torch.cat(all_hln, dim=0)
        hln_valid_mask = (all_hln_tensor[:, 0] != -1)
        all_hln_tensor = all_hln_tensor[hln_valid_mask]
        hln_uni = torch.unique(all_hln_tensor, dim=0).long()

        node_weight = weight_out[hln_uni[:, 0],
        hln_uni[:, 1],
        hln_uni[:, 2],
        hln_uni[:, 3]]
        higher_weight_mask = (node_weight > weight_threshold)
        higher_weight_mask = torch.squeeze(higher_weight_mask)
        refine_node = hln_uni[higher_weight_mask]

        sel = (*refine_node.T,)
        self.RefineCorners(sel=sel)

        self.FindNeighCornerA()

    def AdaptiveResolutionVolSDF(
            self,
            dset_train,
            sdf_threshold: float = 0.0,
            loss_threshold: float = 0.0,
            epoch_id: int = 2):

        # weight_out = torch.zeros(self.child.size(0), self.child.size(1), self.child.size(2), self.child.size(3), 1,
        #                          dtype=torch.float32, device=self.CornerSH.device)
        sdf_out = 999 * torch.ones(self.child.size(0), self.child.size(1), self.child.size(2), self.child.size(3), 1,
                                   dtype=torch.float32, device=self.CornerSH.device)

        chunk_size = 5000

        all_hitnode_out = -1 * torch.ones(0, 4, dtype=torch.int32,
                                          device=self.CornerSH.device)
        for i in tqdm(range(0, dset_train.rays.origins.size(0), chunk_size)):
            chunk_origins = dset_train.rays.origins[i: i + chunk_size]
            chunk_dirs = dset_train.rays.dirs[i: i + chunk_size]

            im_gt = dset_train.rays.gt[i: i + chunk_size]

            rays = Rays(chunk_origins, chunk_dirs)

            hit_stride = self.RayHitNumVOlSDF(rays)

            rgb_out = torch.zeros_like(im_gt)
            hitnode_out = -1 * torch.ones(rays.origins.size(0), hit_stride[0], 4, dtype=torch.int32,
                                          device=self.CornerSH.device)

            _C.volume_render_volsdf_refine_sdf_LOT(self._LOTspecA(),
                                                   rays._to_cpp(),
                                                   self.opt._to_cpp(),
                                                   rgb_out,
                                                   hitnode_out,
                                                   sdf_out)

            mse = ((rgb_out - im_gt) ** 2)
            mse = torch.min(mse, 1)[0]

            higher_loss_mask = (mse > loss_threshold)
            hitnode_out_hl = hitnode_out[higher_loss_mask]
            hitnode_out_hl = hitnode_out_hl.view(-1, 4)
            hitnode_out_valid_mask = (hitnode_out_hl[:, 0] != -1)
            hitnode_out_hl = hitnode_out_hl[hitnode_out_valid_mask]
            all_hitnode_out = torch.cat((all_hitnode_out, hitnode_out_hl), dim=0)
            all_hitnode_out = torch.unique(all_hitnode_out, dim=0)

        all_hitnode_out = all_hitnode_out.long()
        node_sdf = sdf_out[all_hitnode_out[:, 0],
        all_hitnode_out[:, 1],
        all_hitnode_out[:, 2],
        all_hitnode_out[:, 3]]
        higher_sdf_mask = (node_sdf <= sdf_threshold)
        higher_sdf_mask = torch.squeeze(higher_sdf_mask)
        refine_node = all_hitnode_out[higher_sdf_mask]

        sel = (*refine_node.T,)
        self.RefineCorners(sel=sel)

        self.FindNeighCornerA()
        self.FindAllNeighCorner()
        self.FindGhostNeighbors()

        if epoch_id < 1:
            ksize = 5
            self.NodeGaussNeighbors, self.NodeGaussKernals = self.FindGaussianNeigh(ksize=ksize, sigma=0.8)

        ksize = 3
        self.NodeGaussGradNeighbors, self.NodeGaussGradKernals = self.FindGaussianNeigh(ksize=ksize, sigma=0.8)
        self.NodeGaussGradKernals[:, int(ksize * ksize * ksize / 2)] = self.NodeGaussGradKernals[:,
                                                                       int(ksize * ksize * ksize / 2)] - 1.0

    def AdaptiveResolutionColVolSDF(
            self,
            dset_train,
            n_samples: int = 256,
            sdf_threshold: float = 0.0,
            sdf_offset: float = 0.0,
            geo_threshold: float = 0.0,
            max_corner_threshold: int = 3e8,
            is_abs: bool = False,
            dilate_times: int = 1
    ):

        chunk_size = 10000

        node_mse = -1 * 999 * torch.ones(self.child.size(0), self.child.size(1), self.child.size(2), self.child.size(3),
                                         1, dtype=torch.float32, device=self.CornerSH.device)

        for i in tqdm(range(0, dset_train.rays.origins.size(0), chunk_size)):
            chunk_origins = dset_train.rays.origins[i: i + chunk_size]
            chunk_dirs = dset_train.rays.dirs[i: i + chunk_size]

            im_gt = dset_train.rays.gt[i: i + chunk_size]

            rays = Rays(chunk_origins, chunk_dirs)

            rgb_out = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32, device=self.CornerSH.device)

            cuda_tree = self._LOTspecA()
            cuda_opt = self.opt._to_cpp()

            cu_fn = _C.__dict__[f"volume_render_volsdf_test_LOT"]
            #  with utils.Timing("actual_render"):
            cu_fn(
                cuda_tree,
                rays._to_cpp(),
                cuda_opt,
                rgb_out
            )

            ray_mse = ((rgb_out - im_gt) ** 2).mean(dim=1)
            ray_mse = ray_mse.view(-1)

            _C.volume_render_volsdf_colrefine_LOT(
                cuda_tree,
                rays._to_cpp(),
                cuda_opt,
                ray_mse,
                node_mse
            )

        sel = (*self._all_leaves().T,)
        leaf_nodes = torch.stack(sel, dim=-1).to(device=self.CornerSH.device)

        leaf_sdf = self.SampleSDF(leaf_nodes, n_samples, sdf_threshold, sdf_offset, is_abs)

        node_mask = (leaf_sdf <= sdf_threshold)
        select_leaf_nodes = leaf_nodes[node_mask]

        select_leaf_nodes = self.DilateCorners(d_nodes=select_leaf_nodes, dilate_times=dilate_times)

        select_node_mse = node_mse[select_leaf_nodes[:, 0].long(), select_leaf_nodes[:, 1].long(),
        select_leaf_nodes[:, 2].long(), select_leaf_nodes[:, 3].long()]

        select_node_mse, sort_index = torch.sort(select_node_mse, dim=0, descending=True)

        capacity_free = float(max_corner_threshold - self.CornerSH.size(0))

        if capacity_free > 0:
            node_refine_num = int(capacity_free / 7.0)

            if node_refine_num < select_node_mse.size(0):
                c_node_mse = select_node_mse[node_refine_num - 1]
                for i in range(node_refine_num, select_node_mse.size(0), 1):
                    if select_node_mse[i] != c_node_mse:
                        node_refine_num = i
                        break

                sort_index = sort_index.view(-1)
                select_leaf_nodes = select_leaf_nodes[sort_index]
                select_leaf_nodes = select_leaf_nodes[:node_refine_num]

                # select_leaf_nodes = self.DilateCorners(d_nodes=select_leaf_nodes, dilate_times=dilate_times)

            current_depth = self.c_depth
            c_sel = (*select_leaf_nodes.T,)
            while True:
                _, c_sel = self.RefineCorners(repeats=1, sel=c_sel)

                c_nodes = torch.stack(c_sel, dim=-1).to(device=self.CornerSH.device)
                c_node_depths = self.parent_depth[c_nodes[:, 0].long(), 1]
                depth_mask = (c_node_depths <= current_depth)

                c_nodes = c_nodes[depth_mask]

                if c_nodes.numel() > 0:
                    c_sdf = self.SampleSDF(c_nodes, n_samples, sdf_threshold, sdf_offset, is_abs)
                    sdf_mask = (c_sdf <= sdf_threshold)
                    c_nodes = c_nodes[sdf_mask]
                    c_sel = (*c_nodes.T,)
                else:
                    break

            # sel = (*self._all_leaves().T,)
            # leaf_nodes = torch.stack(sel, dim=-1).to(device=self.CornerSH.device)
            # self.FilterGeoCorners(leaf_nodes, geo_threshold, sdf_offset, False, n_samples, dilate_times=dilate_times)

            # self.FindAllNeighCorner()
            # self.FindGhostNeighbors()
            #
            # ksize = 3
            # self.NodeGaussGradNeighbors, self.NodeGaussGradKernals = self.FindCubicGaussianNeigh(sigma=0.8)
            # self.NodeGaussGradKernals[:, int(ksize * ksize * ksize / 2)] = self.NodeGaussGradKernals[:,
            #                                                                int(ksize * ksize * ksize / 2)] - 1.0

    def AdaptiveResolutionGeoVolSDF(
            self,
            n_samples: int = 256,
            sdf_threshold: float = 0.0,
            sdf_offset: float = 0.0,
            geo_threshold: float = 0.0,
            is_abs: bool = False,
            dilate_times: int = 1,
            com_gauss=True):

        sel = (*self._all_leaves().T,)
        leaf_nodes = torch.stack(sel, dim=-1).to(device=self.CornerSH.device)

        leaf_sdf = self.SampleSDF(leaf_nodes, n_samples, sdf_threshold, sdf_offset, is_abs)

        sdf_mask = (leaf_sdf <= sdf_threshold)
        select_leaf_nodes = leaf_nodes[sdf_mask]

        select_leaf_nodes = self.DilateCorners(d_nodes=select_leaf_nodes, dilate_times=dilate_times)

        current_depth = self.c_depth
        c_sel = (*select_leaf_nodes.T,)
        while True:
            _, c_sel = self.RefineCorners(repeats=1, sel=c_sel)

            c_nodes = torch.stack(c_sel, dim=-1).to(device=self.CornerSH.device)
            c_node_depths = self.parent_depth[c_nodes[:, 0].long(), 1]
            depth_mask = (c_node_depths <= current_depth)

            c_nodes = c_nodes[depth_mask]

            if c_nodes.numel() > 0:
                c_sdf = self.SampleSDF(c_nodes, n_samples, sdf_threshold, sdf_offset, is_abs)
                sdf_mask = (c_sdf <= sdf_threshold)
                c_nodes = c_nodes[sdf_mask]
                c_sel = (*c_nodes.T,)
            else:
                break

        sel = (*self._all_leaves().T,)
        leaf_nodes = torch.stack(sel, dim=-1).to(device=self.CornerSH.device)
        self.FilterGeoCorners(leaf_nodes, geo_threshold, sdf_offset, False, n_samples, dilate_times=dilate_times)

        self.FindAllNeighCorner()
        self.FindGhostNeighbors()

        if com_gauss:
            ksize = 3
            self.NodeGaussGradNeighbors, self.NodeGaussGradKernals = self.FindCubicGaussianNeigh(sigma=0.8)
            self.NodeGaussGradKernals[:, int(ksize * ksize * ksize / 2)] = self.NodeGaussGradKernals[:,
                                                                           int(ksize * ksize * ksize / 2)] - 1.0

    def DilateCorners(self,
                      d_nodes,
                      dilate_times):

        unique_d_nodes = torch.unique(d_nodes[:, 0], dim=0, sorted=False)

        all_d_nodes = []
        for i in range(int(dilate_times)):

            for i in range(8):
                if i == 0:
                    c_index = torch.tensor([0, 0, 0], dtype=torch.int64, device=self.CornerSH.device)
                elif i == 1:
                    c_index = torch.tensor([0, 0, 1], dtype=torch.int64, device=self.CornerSH.device)
                elif i == 2:
                    c_index = torch.tensor([0, 1, 0], dtype=torch.int64, device=self.CornerSH.device)
                elif i == 3:
                    c_index = torch.tensor([0, 1, 1], dtype=torch.int64, device=self.CornerSH.device)
                elif i == 4:
                    c_index = torch.tensor([1, 0, 0], dtype=torch.int64, device=self.CornerSH.device)
                elif i == 5:
                    c_index = torch.tensor([1, 0, 1], dtype=torch.int64, device=self.CornerSH.device)
                elif i == 6:
                    c_index = torch.tensor([1, 1, 0], dtype=torch.int64, device=self.CornerSH.device)
                elif i == 7:
                    c_index = torch.tensor([1, 1, 1], dtype=torch.int64, device=self.CornerSH.device)

                c_nodes = torch.zeros(unique_d_nodes.size(0), 4, dtype=torch.int64,
                                      device=self.CornerSH.device)

                c_nodes[:, 0] = unique_d_nodes
                c_nodes[:, 1] = c_index[0]
                c_nodes[:, 2] = c_index[1]
                c_nodes[:, 3] = c_index[2]

                leaves_node_mask = (self.child[c_nodes[:, 0], c_nodes[:, 1],
                c_nodes[:, 2], c_nodes[:, 3]] == 0)

                c_node = c_nodes[leaves_node_mask]
                all_d_nodes.append(c_node)
        all_d_nodes = torch.cat(all_d_nodes, dim=0)

        return all_d_nodes

    def FilterGeoCorners(self,
                         wait_filter_nodes,
                         sdf_threshold,
                         sdf_offset,
                         is_abs,
                         n_samples,
                         dilate_times):

        node_sdf = self.SampleSDF(wait_filter_nodes, n_samples, sdf_threshold, sdf_offset, is_abs)

        lower_sdf_mask = (node_sdf <= sdf_threshold)
        filter_nodes = wait_filter_nodes[lower_sdf_mask.squeeze()]

        all_filter_nodes = self.DilateCorners(d_nodes=filter_nodes, dilate_times=dilate_times)

        self.ValidGeoCorner = self.NodeCorners[all_filter_nodes[:, 0], all_filter_nodes[:, 1],
        all_filter_nodes[:, 2], all_filter_nodes[:, 3]]
        self.ValidGeoCorner = self.ValidGeoCorner.view(-1)
        self.ValidGeoCorner = torch.unique(self.ValidGeoCorner, dim=0, sorted=False)

        self.SetGeoCorner(corner=self.ValidGeoCorner)

    def SampleSDF(self, nodes, n_samples, sdf_thresh, sdf_offset, is_abs):
        chunk_size = 20000

        all_nodes_sdf = []
        for i in tqdm(range(0, nodes.size(0), chunk_size)):
            c_nodes = nodes[i:i + chunk_size]

            l_node_corners = self.CalCorner(c_nodes)
            node_depths = self.parent_depth[c_nodes[:, 0].long(), 1]
            node_lengths = 1.0 / (2 ** (node_depths + 1))

            offsets = torch.rand((l_node_corners.shape[0], n_samples, 3), device=self.CornerSH.device,
                                 dtype=torch.float32)
            offsets = node_lengths[:, None, None] * offsets

            sample_pos = l_node_corners[:, None] + offsets

            node_corners = self.NodeCorners[c_nodes[:, 0], c_nodes[:, 1],
            c_nodes[:, 2], c_nodes[:, 3]]

            rand_cells = torch.arange(0, sample_pos.size(0), dtype=torch.int32, device=
            self.CornerSDF.device)

            sample_data = 999 * torch.ones(sample_pos.size(0), 1, dtype=torch.float32, device=self.CornerSDF.device)

            _C.sample_tri_min_interp(
                node_corners,
                sample_pos,
                l_node_corners,
                node_lengths,
                self.CornerSDF,
                rand_cells,
                sdf_thresh, sdf_offset,
                is_abs,
                0, 1,
                sample_data
            )

            sample_data = sample_data.view(-1)

            all_nodes_sdf.append(sample_data)

        all_nodes_sdf = torch.cat(all_nodes_sdf, dim=0)

        # sdf_out = 999 * torch.ones(self.child.size(0), self.child.size(1), self.child.size(2), self.child.size(3), 1,
        #                            dtype=torch.float32, device=self.CornerSH.device)
        # sdf_out[leaf_node[:, 0].long(), leaf_node[:, 1].long(), leaf_node[:, 2].long(), leaf_node[:,
        #                                                                                 3].long(), 0] = all_leaves_sdf

        return all_nodes_sdf

    def SampleTriCuda(self, nodes, n_samples, mode):
        chunk_size = 10000

        all_nodes_data = []
        for i in tqdm(range(0, nodes.size(0), chunk_size)):
            c_nodes = nodes[i:i + chunk_size]

            l_node_corners = self.CalCorner(c_nodes)
            node_depths = self.parent_depth[c_nodes[:, 0].long(), 1]
            node_lengths = 1.0 / (2 ** (node_depths + 1))

            offsets = torch.rand((l_node_corners.shape[0], n_samples, 3), device=self.CornerSH.device,
                                 dtype=torch.float32)
            offsets = node_lengths[:, None, None] * offsets

            sample_pos = l_node_corners[:, None] + offsets

            node_corners = self.NodeCorners[c_nodes[:, 0], c_nodes[:, 1],
            c_nodes[:, 2], c_nodes[:, 3]]

            rand_cells = torch.arange(0, sample_pos.size(0), dtype=torch.int32, device=
            self.CornerSDF.device)

            if mode == 'sdf':
                sample_data = torch.zeros(sample_pos.size(0), 1, dtype=torch.float32, device=self.CornerSDF.device)
                _C.sample_tri_interp(
                    node_corners,
                    sample_pos,
                    l_node_corners,
                    node_lengths,
                    self.CornerSDF,
                    rand_cells,
                    0, 1,
                    sample_data
                )
            else:
                sample_data = torch.zeros(sample_pos.size(0), self.CornerSH.size(1), dtype=torch.float32,
                                          device=self.CornerSDF.device)
                _C.sample_tri_interp(
                    node_corners,
                    sample_pos,
                    l_node_corners,
                    node_lengths,
                    self.CornerSH,
                    rand_cells,
                    0, self.CornerSH.size(1),
                    sample_data
                )

            all_nodes_data.append(sample_data)

        all_nodes_data = torch.cat(all_nodes_data, dim=0)

        # sdf_out = 999 * torch.ones(self.child.size(0), self.child.size(1), self.child.size(2), self.child.size(3), 1,
        #                            dtype=torch.float32, device=self.CornerSH.device)
        # sdf_out[leaf_node[:, 0].long(), leaf_node[:, 1].long(), leaf_node[:, 2].long(), leaf_node[:,
        #                                                                                 3].long(), 0] = all_leaves_sdf

        return all_nodes_data

    def SampleSH(self, nodes, n_samples, mode):
        chunk_size = 5000

        all_nodes_sh = []
        for i in tqdm(range(0, nodes.size(0), chunk_size)):
            c_nodes = nodes[i:i + chunk_size]

            l_node_corners = self.CalCorner(c_nodes)
            node_depths = self.parent_depth[c_nodes[:, 0].long(), 1]
            node_lengths = 1.0 / (2 ** (node_depths + 1))

            offsets = torch.rand((l_node_corners.shape[0], n_samples, 3), device=self.CornerSH.device,
                                 dtype=torch.float32)
            offsets = node_lengths[:, None, None] * offsets

            pos = l_node_corners[:, None] + offsets
            pos = pos.view(-1, 3)

            treeview = self[LocalIndex(pos)]
            c_leaf_node = torch.stack(treeview.key, dim=-1)

            node_corners = self.NodeCorners[c_leaf_node[:, 0], c_leaf_node[:, 1],
            c_leaf_node[:, 2], c_leaf_node[:, 3]]
            node_num = node_corners.shape[0]
            node_corners = node_corners.view([node_num * 8])
            nc_sh = self.CornerSH[node_corners.long()]
            nc_sh = nc_sh.view([node_num, 8, -1])

            cube_sz = treeview.lengths_local
            low_pos = treeview.corners_local

            treeview = None

            sh_t = self.TriInterpSH(nc_sh, pos, low_pos, cube_sz)
            sh_t = sh_t.view(-1, n_samples)

            if mode == 'min':
                sh_t = torch.min(sh_t, dim=1)[0]
            else:
                sh_t = torch.mean(sh_t, dim=1)

            all_nodes_sh.append(sh_t)
        all_nodes_sh = torch.cat(all_nodes_sh, dim=0)

        # sdf_out = 999 * torch.ones(self.child.size(0), self.child.size(1), self.child.size(2), self.child.size(3), 1,
        #                            dtype=torch.float32, device=self.CornerSH.device)
        # sdf_out[leaf_node[:, 0].long(), leaf_node[:, 1].long(), leaf_node[:, 2].long(), leaf_node[:,
        #                                                                                 3].long(), 0] = all_leaves_sdf

        return all_nodes_sh

    def BoundOctree(self,
                    n_samples: int = 256,
                    sdf_thresh: float = 0.0,
                    sdf_offset: float = 0.0,
                    bbox_scale: float = 1.2):

        sel = (*self._all_leaves().T,)
        leaf_nodes = torch.stack(sel, dim=-1).to(device=self.CornerSH.device)

        leaf_sdf = self.SampleSDF(leaf_nodes, n_samples, sdf_thresh, sdf_offset, is_abs=False)

        sdf_mask = (leaf_sdf <= sdf_thresh)
        select_leaf_nodes = leaf_nodes[sdf_mask]

        occupied_node_corners = self.NodeCorners[
            select_leaf_nodes[:, 0].long(),
            select_leaf_nodes[:, 1].long(),
            select_leaf_nodes[:, 2].long(),
            select_leaf_nodes[:, 3].long()]

        occupied_node_corners = occupied_node_corners.view(-1)
        occupied_node_corners = torch.unique(occupied_node_corners, dim=0)

        inverse_dict = {v: k for k, v in self.CornerDict.items()}
        occupied_corner_pos = self.FromCornerIndexGetPos(occupied_node_corners, inverse_dict)
        occupied_corner_pos = occupied_corner_pos.type(torch.float32)
        occupied_corner_pos /= (2 ** (self.depth_limit + 1))

        lowest_corner = occupied_corner_pos.min(dim=0)[0]
        highest_corner = occupied_corner_pos.max(dim=0)[0]
        bbox_length = highest_corner - lowest_corner

        # offset = (bbox_scale - 1.0) * bbox_length / 2.0
        # lowest_corner -= offset
        # highest_corner += offset
        # for i in range(lowest_corner.size(0)):
        #     if lowest_corner[i] < 0.0:
        #         lowest_corner[i] = 0.02
        # for i in range(highest_corner.size(0)):
        #     if highest_corner[i] > 1.0:
        #         highest_corner[i] = 0.98
        #
        # bbox_length = highest_corner - lowest_corner

        occupied_corner_pos[:, 0] = (occupied_corner_pos[:, 0] - self.offset[0]) / self.invradius[0]
        occupied_corner_pos[:, 1] = (occupied_corner_pos[:, 1] - self.offset[1]) / self.invradius[1]
        occupied_corner_pos[:, 2] = (occupied_corner_pos[:, 2] - self.offset[2]) / self.invradius[2]

        lc = occupied_corner_pos.min(dim=0)[0] - 0.5 / (2 ** (self.depth_limit + 1))
        uc = occupied_corner_pos.max(dim=0)[0] + 0.5 / (2 ** (self.depth_limit + 1))

        center = (lc + uc) * 0.5
        radius = (uc - lc) * 0.5

        return center, radius, lowest_corner, bbox_length

    def ComDataForAnoTree(self,
                          points,
                          bbox_length,
                          bbox_corner,
                          com_sh=False):
        points_to_t = torch.zeros_like(points)
        points_to_t[:, 0] = bbox_corner[0] + bbox_length[0] * points[:, 0]
        points_to_t[:, 1] = bbox_corner[1] + bbox_length[1] * points[:, 1]
        points_to_t[:, 2] = bbox_corner[2] + bbox_length[2] * points[:, 2]

        chunk_size = 20000

        all_corners_sdf = []
        all_corners_sh = []
        for i in tqdm(range(0, points_to_t.size(0), chunk_size)):
            c_corners_to_t = points_to_t[i:i + chunk_size]

            treeview = self[LocalIndex(c_corners_to_t)]
            leaf_node = torch.stack(treeview.key, dim=-1)

            node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
            leaf_node[:, 2], leaf_node[:, 3]]
            node_num = node_corners.shape[0]
            node_corners = node_corners.view([node_num * 8])

            nc_data = self.CornerSDF[node_corners.long()]
            nc_data = nc_data.view([node_num, 8])

            cube_sz = treeview.lengths_local
            low_pos = treeview.corners_local

            new_sdf_t = self.TriInterpS(nc_data, c_corners_to_t, low_pos, cube_sz)
            all_corners_sdf.append(new_sdf_t)
            #######################################################
            if com_sh:
                nc_data = self.CornerSH[node_corners.long()]
                nc_data = nc_data.view([node_num, 8, -1])

                new_sh_t = self.TriInterpSH(nc_data, c_corners_to_t, low_pos, cube_sz)
                all_corners_sh.append(new_sh_t)
            else:
                all_corners_sh = None

        all_corners_sdf = torch.cat(all_corners_sdf, dim=0)

        if com_sh:
            all_corners_sh = torch.cat(all_corners_sh, dim=0)

        return all_corners_sdf, all_corners_sh

    def AddTVDensityGrad(self, grad: torch.Tensor,
                         scaling: float = 1.0,
                         sparse_frac: float = 0.01,
                         logalpha: bool = False,
                         contiguous: bool = True
                         ):

        rand_cells = self.GetRandCells(sparse_frac, contiguous=contiguous)

        _C.tv_grad_sparse_LOT(self.NodeNeighbors, self.CornerD,
                              rand_cells,
                              0, 1, scaling,
                              logalpha,
                              grad)

    def AddTVSDFGrad(self, grad: torch.Tensor,
                     scaling: float = 1.0,
                     sparse_frac: float = 0.01,
                     logalpha: bool = False,
                     contiguous: bool = True
                     ):

        # rand_cells = self.GetRandCells(sparse_frac, contiguous=contiguous)
        rand_cells = torch.arange(0, self.CornerSDF.size(0), dtype=torch.int32, device=
        self.CornerSDF.device)

        _C.tv_grad_sparse_LOT(self.NodeNeighbors, self.CornerSDF,
                              rand_cells,
                              0, 1, scaling,
                              logalpha,
                              grad)

    def AddTVThirdordSDFGrad(self, grad: torch.Tensor,
                             scaling: float = 1.0,
                             sparse_frac: float = 0.01,
                             contiguous: bool = True
                             ):

        rand_cells = self.GetRandCells(sparse_frac, contiguous=contiguous)
        # rand_cells = self.ValidGeoCorner

        _C.tv_grad_sparse_thirdord_mid_LOT(self.NodeAllNeighbors,
                                           self.NodeAllNeighLen,
                                           self.NodeGhoNeighbors,
                                           self.NodeGhoCoeff,
                                           self.CornerMap,
                                           self.CornerSDF,
                                           rand_cells,
                                           0, 1, scaling,
                                           grad)

    def AddTVColorGrad(
            self,
            grad: torch.Tensor,
            start_dim: int = 0,
            end_dim: Optional[int] = None,
            scaling: float = 1.0,
            sparse_frac: float = 0.01,
            logalpha: bool = False,
            contiguous: bool = True
    ):
        if end_dim is None:
            end_dim = self.CornerSH.size(1)
        end_dim = end_dim + self.CornerSH.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.CornerSH.size(1) if start_dim < 0 else start_dim

        rand_cells = self.GetRandCells(sparse_frac, contiguous=contiguous)

        _C.tv_grad_sparse_LOT(self.NodeNeighbors, self.CornerSH,
                              rand_cells,
                              start_dim, end_dim, scaling,
                              logalpha,
                              grad)

    def AddTVThirdordColorGrad(
            self,
            grad: torch.Tensor,
            start_dim: int = 0,
            end_dim: Optional[int] = None,
            scaling: float = 1.0,
            sparse_frac: float = 0.01,
            contiguous: bool = True
    ):
        if end_dim is None:
            end_dim = self.CornerSH.size(1)
        end_dim = end_dim + self.CornerSH.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.CornerSH.size(1) if start_dim < 0 else start_dim

        rand_cells = self.GetRandCells(sparse_frac, contiguous=contiguous)

        _C.tv_grad_sparse_thirdord_mid_LOT(self.NodeAllNeighbors,
                                           self.NodeAllNeighLen,
                                           self.NodeGhoNeighbors,
                                           self.NodeGhoCoeff,
                                           self.CornerMap,
                                           self.CornerSH,
                                           rand_cells,
                                           start_dim, end_dim, scaling,
                                           grad)

    def GetRandCells(self, sparse_frac: float, force: bool = False, contiguous: bool = True):
        if sparse_frac < 1.0 or force:
            current_size = self.ValidGeoCorner.size(0)
            sparse_num = max(int(sparse_frac * current_size), 1)
            if contiguous:
                start = np.random.randint(0, current_size)
                arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=
                self.CornerIndex.device)

                if start > current_size - sparse_num:
                    arr[current_size - sparse_num - start:] -= current_size

                rand_cell = self.ValidGeoCorner[arr.long()]

                return rand_cell
            else:
                arr = torch.randint(0, current_size, (sparse_num,), dtype=torch.int32, device=
                self.CornerIndex.device)

                rand_cell = self.ValidGeoCorner[arr.long()]
                return rand_cell

        return None

    def GetDataGrads(self):
        ret = []
        for subitem in ["CornerD", "CornerSH"]:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (
                        not hasattr(param, "grad")
                        or param.grad is None
                        or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def GetDataGradsSDF(self):
        ret = []
        for subitem in ["CornerSDF", "CornerSH", "LearnS"]:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (
                        not hasattr(param, "grad")
                        or param.grad is None
                        or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def GetDataGradsVolSDF(self):
        ret = []
        for subitem in ["CornerSDF", "CornerSH", "Beta", "LearnS"]:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (
                        not hasattr(param, "grad")
                        or param.grad is None
                        or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def OptimDensity(self, lr: float, beta: float = 0.9, epsilon: float = 1e-8,
                     optim: str = 'rmsprop'):
        if (self.density_rms is None or self.density_rms.shape != self.CornerD.shape):
            del self.density_rms
            self.density_rms = torch.zeros_like(self.CornerD.data)  # FIXME init?
        _C.rmsprop_step_LOT(
            self.CornerD.data,
            self.density_rms,
            self.CornerD.grad,
            beta,
            lr,
            epsilon,
            -1e9,
            lr
        )

    def OptimSDF(self, lr: float, beta: float = 0.9, epsilon: float = 1e-8,
                 optim: str = 'rmsprop'):
        if (self.sdf_rms is None or self.sdf_rms.shape != self.CornerSDF.shape):
            del self.sdf_rms
            self.sdf_rms = torch.zeros_like(self.CornerSDF.data)  # FIXME init?
        _C.rmsprop_step_LOT(
            self.CornerSDF.data,
            self.sdf_rms,
            self.CornerSDF.grad,
            beta,
            lr,
            epsilon,
            -1e9,
            lr
        )

    def OptimData(self,
                  lr_sh: float,
                  lr_sdf: float,
                  beta: float = 0.9,
                  epsilon: float = 1e-8,
                  leaf_nodes: torch.tensor = None):
        data_sh = self.data.data[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3], 0:27]
        data_sh_grad = self.data.grad.data[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3], 0:27]

        if (self.data_sh_rms is None or self.data_sh_rms.shape != data_sh.shape):
            del self.data_sh_rms
            self.data_sh_rms = torch.zeros_like(data_sh)  # FIXME init?
        _C.rmsprop_step_LOT(
            data_sh,
            self.data_sh_rms,
            data_sh_grad,
            beta,
            lr_sh,
            epsilon,
            -1e9,
            lr_sh
        )

        data_sdf = self.data.data[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3], 27:]
        data_sdf_grad = self.data.grad.data[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3], 27:]

        if (self.data_sdf_rms is None or self.data_sdf_rms.shape != data_sdf.shape):
            del self.data_sdf_rms
            self.data_sdf_rms = torch.zeros_like(data_sdf)  # FIXME init?
        _C.rmsprop_step_LOT(
            data_sdf,
            self.data_sdf_rms,
            data_sdf_grad,
            beta,
            lr_sdf,
            epsilon,
            -1e9,
            lr_sdf
        )

        self.data.data[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3], 0:27] = data_sh
        self.data.data[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3], 27] = data_sdf[:, 0]

    def OptimBeta(self, lr: float, beta: float = 0.9, epsilon: float = 1e-8,
                  optim: str = 'rmsprop'):
        if (self.beta_rms is None or self.beta_rms.shape != self.Beta.shape):
            del self.beta_rms
            self.beta_rms = torch.zeros_like(self.Beta.data)  # FIXME init?
        _C.rmsprop_step_LOT(
            self.Beta.data,
            self.beta_rms,
            self.Beta.grad,
            beta,
            lr,
            epsilon,
            -1e9,
            lr
        )

    def OptimSH(self, lr: float, beta: float = 0.9, epsilon: float = 1e-8,
                optim: str = 'rmsprop'):

        if self.sh_rms is None or self.sh_rms.shape != self.CornerSH.shape:
            del self.sh_rms
            self.sh_rms = torch.zeros_like(self.CornerSH.data)  # FIXME init?
        _C.rmsprop_step_LOT(
            self.CornerSH.data,
            self.sh_rms,
            self.CornerSH.grad,
            beta,
            lr,
            epsilon,
            -1e9,
            lr
        )

    def ExtractGeometry(self, resolution, bbox_corner, bbox_length, base_exp_dir, iter_step):
        N = 64
        X = torch.linspace(0, 1, resolution).split(N)
        Y = torch.linspace(0, 1, resolution).split(N)
        Z = torch.linspace(0, 1, resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pos = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        pos = torch.tensor(pos, dtype=torch.float32, device=self.CornerSH.device)

                        treeview = self[LocalIndex(pos)]
                        leaf_node = torch.stack(treeview.key, dim=-1)

                        node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
                        leaf_node[:, 2], leaf_node[:, 3]]
                        node_num = node_corners.shape[0]
                        node_corners = node_corners.view([node_num * 8])
                        nc_sdf = self.CornerSDF[node_corners.long()]
                        nc_sdf = nc_sdf.view([node_num, 8])

                        cube_sz = treeview.lengths_local
                        low_pos = treeview.corners_local

                        sdf_t = self.TriInterpS(nc_sdf, pos, low_pos, cube_sz)

                        val = sdf_t.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

        vertices, triangles = mcubes.marching_cubes(u, 0)
        bbox_corner_np = bbox_corner.detach().cpu().numpy()
        bbox_length_np = bbox_length.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * bbox_length_np[None, :] + bbox_corner_np[None, :]

        os.makedirs(os.path.join(base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(base_exp_dir, 'meshes', '{:0>8d}.ply'.format(iter_step)))

    def ExtractGaussGeometry(self, resolution, bound_min, bound_max, base_exp_dir, iter_step, ksize):
        self.CornerGaussSDF.data = torch.zeros(self.CornerSDF.size(0), 1, dtype=torch.float32, device=
        self.CornerSDF.device)

        _C.gaussian_sdf_conv_LOT(self.NodeGaussNeighbors,
                                 self.NodeGaussKernals,
                                 self.CornerMap,
                                 self.CornerSDF,
                                 self.ValidGeoCorner,
                                 0, ksize * ksize * ksize,
                                 self.CornerGaussSDF)

        N = 64
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pos = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        pos = torch.tensor(pos, dtype=torch.float32, device=self.CornerSH.device)

                        treeview = self[LocalIndex(pos)]
                        leaf_node = torch.stack(treeview.key, dim=-1)

                        node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
                        leaf_node[:, 2], leaf_node[:, 3]]
                        node_num = node_corners.shape[0]
                        node_corners = node_corners.view([node_num * 8])
                        nc_sdf = self.CornerGaussSDF[node_corners.long()]
                        nc_sdf = nc_sdf.view([node_num, 8])

                        cube_sz = treeview.lengths_local
                        low_pos = treeview.corners_local

                        sdf_t = self.TriInterpS(nc_sdf, pos, low_pos, cube_sz)

                        val = sdf_t.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

        vertices, triangles = mcubes.marching_cubes(u, 0)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

        os.makedirs(os.path.join(base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(base_exp_dir, 'meshes', '{:0>8d}_gauss.ply'.format(iter_step)))

    def RenderSDFIsolines(self, z_pos, radius, vis_dir, epoch_id):
        res_x = 1000
        radio = radius[1] / radius[2]
        step_y = 1.0 / (res_x * radio)

        yy, xx = torch.meshgrid(
            torch.arange(start=0.0, end=1.0, step=1e-3, dtype=torch.float32, device=self.CornerSDF.device),
            torch.arange(start=0.0, end=1.0, step=step_y, dtype=torch.float32, device=self.CornerSDF.device),
        )
        zz = torch.ones_like(xx) * z_pos

        res_y = xx.size(1)

        pos = torch.stack((zz, xx, yy), dim=-1)
        pos = pos.view(-1, 3)

        treeview = self[LocalIndex(pos)]
        leaf_node = torch.stack(treeview.key, dim=-1)

        node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
        leaf_node[:, 2], leaf_node[:, 3]]
        node_num = node_corners.shape[0]
        node_corners = node_corners.view([node_num * 8])
        nc_sdf = self.CornerSDF[node_corners.long()]
        nc_sdf = nc_sdf.view([node_num, 8])

        cube_sz = treeview.lengths_local
        low_pos = treeview.corners_local

        sdf_t = self.TriInterpS(nc_sdf, pos, low_pos, cube_sz)[:, 0].cpu().detach().numpy()

        vmax = 0.2
        vmin = -vmax
        isoline_spacing = 0.03
        isoline_thickness = 0.1

        mi.set_variant('scalar_rgb')

        sdf_color = cm.coolwarm(plt.Normalize(vmin, vmax)(np.array(sdf_t)))[..., :3] ** 2.2

        select_mask = (np.abs(sdf_t) / isoline_spacing) - np.floor(
            np.abs(sdf_t) / isoline_spacing) < isoline_thickness
        sdf_color[select_mask] = 0.2 * sdf_color[select_mask]

        sdf_color = sdf_color.reshape(res_x, res_y, 3)
        vis = (sdf_color * 255).astype(np.uint8)

        imageio.imwrite(f"{vis_dir}/sdf_isolines_{epoch_id:04}.png", vis)

    def RenderSHIsolines(self, z_pos, radius, vis_dir, epoch_id):
        res_x = 1000
        radio = radius[1] / radius[2]
        step_y = 1.0 / (res_x * radio)

        yy, xx = torch.meshgrid(
            torch.arange(start=0.0, end=1.0, step=1e-3, dtype=torch.float32, device=self.CornerSDF.device),
            torch.arange(start=0.0, end=1.0, step=step_y, dtype=torch.float32, device=self.CornerSDF.device),
        )
        zz = torch.ones_like(xx) * z_pos

        res_y = xx.size(1)

        pos = torch.stack((zz, xx, yy), dim=-1)
        pos = pos.view(-1, 3)

        treeview = self[LocalIndex(pos)]
        leaf_node = torch.stack(treeview.key, dim=-1)

        node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
        leaf_node[:, 2], leaf_node[:, 3]]
        node_num = node_corners.shape[0]
        node_corners = node_corners.view([node_num * 8])
        nc_sdf = self.CornerSH[node_corners.long(), 0]
        nc_sdf = nc_sdf.view([node_num, 8])

        cube_sz = treeview.lengths_local
        low_pos = treeview.corners_local

        sdf_t = self.TriInterpS(nc_sdf, pos, low_pos, cube_sz)[:, 0].cpu().detach().numpy()

        vmax = 0.2
        vmin = -vmax
        isoline_spacing = 0.03
        isoline_thickness = 0.1

        mi.set_variant('scalar_rgb')

        sdf_color = cm.coolwarm(plt.Normalize(vmin, vmax)(np.array(sdf_t)))[..., :3] ** 2.2

        select_mask = (np.abs(sdf_t) / isoline_spacing) - np.floor(
            np.abs(sdf_t) / isoline_spacing) < isoline_thickness
        sdf_color[select_mask] = 0.2 * sdf_color[select_mask]

        sdf_color = sdf_color.reshape(res_x, res_y, 3)
        vis = (sdf_color * 255).astype(np.uint8)

        imageio.imwrite(f"{vis_dir}/sh_isolines_{epoch_id:04}.png", vis)

    def RenderGaussSDFIsolines(self, z_pos, vis_dir):
        yy, xx = torch.meshgrid(
            torch.arange(start=0.0, end=1.0, step=1e-3, dtype=torch.float32, device=self.CornerSDF.device),
            torch.arange(start=0.0, end=1.0, step=1e-3, dtype=torch.float32, device=self.CornerSDF.device),
        )
        zz = torch.ones_like(xx) * z_pos
        pos = torch.stack((zz, xx, yy), dim=-1)
        pos = pos.view(-1, 3)

        treeview = self[LocalIndex(pos)]
        leaf_node = torch.stack(treeview.key, dim=-1)

        node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
        leaf_node[:, 2], leaf_node[:, 3]]
        node_num = node_corners.shape[0]
        node_corners = node_corners.view([node_num * 8])
        nc_sdf = self.CornerGaussSDF[node_corners.long()]
        nc_sdf = nc_sdf.view([node_num, 8])

        cube_sz = treeview.lengths_local
        low_pos = treeview.corners_local

        sdf_t = self.TriInterpS(nc_sdf, pos, low_pos, cube_sz)[:, 0].cpu().detach().numpy()

        vmax = 0.2
        vmin = -vmax
        isoline_spacing = 0.03
        isoline_thickness = 0.1

        mi.set_variant('scalar_rgb')

        sdf_color = cm.coolwarm(plt.Normalize(vmin, vmax)(np.array(sdf_t)))[..., :3] ** 2.2

        select_mask = (np.abs(sdf_t) / isoline_spacing) - np.floor(
            np.abs(sdf_t) / isoline_spacing) < isoline_thickness
        sdf_color[select_mask] = 0.2 * sdf_color[select_mask]

        res = 1000

        sdf_color = sdf_color.reshape(res, res, 3)
        vis = (sdf_color * 255).astype(np.uint8)

        imageio.imwrite(f"{vis_dir}/sdf_isolines_gauss.png", vis)

    def RenderSDFIsolinesTest(self, z_pos, vis_dir):
        corner_gradient = torch.zeros(self.CornerSDF.size(0), 3, dtype=torch.float32, device=
        self.CornerSDF.device)

        comp_cells = torch.arange(0, self.CornerSDF.size(0), dtype=torch.int32, device=
        self.CornerSDF.device)

        _C.com_corner_gradient_thirdord_LOT(self.NodeAllNeighbors, self.NodeAllNeighLen,
                                            self.NodeGhoNeighbors, self.NodeGhoCoeff,
                                            self.CornerSDF,
                                            comp_cells,
                                            corner_gradient)

        yy, xx = torch.meshgrid(
            torch.arange(start=0.0, end=1.0, step=1e-3, dtype=torch.float32, device=self.CornerSDF.device),
            torch.arange(start=0.0, end=1.0, step=1e-3, dtype=torch.float32, device=self.CornerSDF.device),
        )
        zz = torch.ones_like(xx) * z_pos
        pos = torch.stack((zz, xx, yy), dim=-1)
        pos = pos.view(-1, 3)

        treeview = self[LocalIndex(pos)]
        leaf_node = torch.stack(treeview.key, dim=-1)

        node_corners = self.NodeCorners[leaf_node[:, 0], leaf_node[:, 1],
        leaf_node[:, 2], leaf_node[:, 3]]
        node_num = node_corners.shape[0]
        node_corners = node_corners.view([node_num * 8])

        for i in range(3):
            nc_sdf = corner_gradient[node_corners.long(), i]
            nc_sdf = nc_sdf.view([node_num, 8])

            cube_sz = treeview.lengths_local
            low_pos = treeview.corners_local

            sdf_t = self.TriInterpS(nc_sdf, pos, low_pos, cube_sz)[:, 0].cpu().detach().numpy()

            vmax = 0.2
            vmin = -vmax
            isoline_spacing = 0.03
            isoline_thickness = 0.1

            mi.set_variant('scalar_rgb')

            sdf_color = cm.coolwarm(plt.Normalize(vmin, vmax)(np.array(sdf_t)))[..., :3] ** 2.2

            select_mask = (np.abs(sdf_t) / isoline_spacing) - np.floor(
                np.abs(sdf_t) / isoline_spacing) < isoline_thickness
            sdf_color[select_mask] = 0.2 * sdf_color[select_mask]

            res = 1000

            sdf_color = sdf_color.reshape(res, res, 3)
            vis = (sdf_color * 255).astype(np.uint8)

            imageio.imwrite(f"{vis_dir}/sdf_isolines_{i}.png", vis)

    def _LOTspecA(self):
        """
        Generate object to pass to C++
        """
        tree_spec = _C.TreeSpecLOT()

        tree_spec.CornerSH = self.CornerSH
        tree_spec.CornerD = self.CornerD
        tree_spec.CornerSDF = self.CornerSDF
        tree_spec.data = self.data
        tree_spec.CornerGaussSDF = self.CornerGaussSDF
        tree_spec.LearnS = self.LearnS
        tree_spec.Beta = self.Beta
        tree_spec.NodeCorners = self.NodeCorners
        tree_spec.NodeAllNeighbors = self.NodeAllNeighbors
        tree_spec.NodeAllNeighLen = self.NodeAllNeighLen
        tree_spec.child = self.child

        tree_spec._offset = self.offset
        tree_spec._scaling = self.invradius

        tree_spec.basis_dim = self.basis_dim
        tree_spec.basis_type = self.basis_type

        return tree_spec

    def _LOTspec(self, world=True):
        """
        Pack tree into a TreeSpec (for passing data to C++ extension)
        """
        tree_spec = _C.TreeSpecLOT()
        tree_spec.CornerSH = self.CornerSH
        tree_spec.CornerD = self.CornerD
        tree_spec.NodeCorners = self.NodeCorners
        tree_spec.child = self.child
        tree_spec.parent_depth = self.parent_depth
        tree_spec.extra_data = self.extra_data if self.extra_data is not None else \
            torch.empty((0, 0), dtype=self.data.dtype, device=self.data.device)
        tree_spec.offset = self.offset if world else torch.tensor(
            [0.0, 0.0, 0.0], dtype=self.data.dtype, device=self.data.device)
        tree_spec.scaling = self.invradius if world else torch.tensor(
            [1.0, 1.0, 1.0], dtype=self.data.dtype, device=self.data.device)
        if hasattr(self, '_weight_accum'):
            tree_spec._weight_accum = self._weight_accum if \
                self._weight_accum is not None else torch.empty(
                0, dtype=self.data.dtype, device=self.data.device)
            tree_spec._weight_accum_max = (self._weight_accum_op == 'max')
        return tree_spec

    def LOTClone(self, device=None):
        return self.LOTPartial(device=device)

        # Persistence

    def LOTSave(self, path, path_d, out_full=True, compress=False):
        """
        Save to from npz file

        :param path: npz path
        :param shrink: if True (default), applies shrink_to_fit before saving
        :param compress: whether to compress the npz; may be slow

        """
        # if shrink:
        # self.shrink_to_fit()
        if out_full:
            data = {
                "data_dim": self.data_dim,
                "basis_dim": self.basis_dim,
                "child": self.child.cpu(),
                "parent_depth": self.parent_depth.cpu(),
                "n_internal": self._n_internal.cpu().item(),
                "n_free": self._n_free.cpu().item(),
                "invradius3": self.invradius.cpu(),
                "offset": self.offset.cpu(),
                "depth_limit": self.depth_limit,
                "geom_resize_fact": self.geom_resize_fact,
                "data": self.data.data.half().cpu().numpy(),  # save CPU Memory
                "NodeCorners": self.NodeCorners.cpu(),
                "NodeAllNeighbors": self.NodeAllNeighbors.cpu(),
                "NodeAllNeighLen": self.NodeAllNeighLen.cpu(),
                "NodeGaussGradNeighbors": self.NodeGaussGradNeighbors.cpu(),
                "NodeGaussGradKernals": self.NodeGaussGradKernals.cpu(),
                "NodeGhoNeighbors": self.NodeGhoNeighbors.cpu(),
                "NodeGhoCoeff": self.NodeGhoCoeff.cpu(),
                "CornerMap": self.CornerMap.cpu(),
                "ValidGeoCorner": self.ValidGeoCorner.cpu(),
                "ValidGeoCornerCoord": self.ValidGeoCornerCoord.cpu(),
                "CornerSDF": self.CornerSDF.data.half().cpu().numpy(),
                "CornerSH": self.CornerSH.data.half().cpu().numpy(),
                "Beta": self.Beta.data.half().cpu().numpy(),
                "n_corner": self._n_corners.cpu().item(),
                "c_depth": self._c_depth.cpu().item()
            }
        else:
            data = {
                "data_dim": self.data_dim,
                "basis_dim": self.basis_dim,
                "child": self.child.cpu(),
                "parent_depth": self.parent_depth.cpu(),
                "n_internal": self._n_internal.cpu().item(),
                "n_free": self._n_free.cpu().item(),
                "invradius3": self.invradius.cpu(),
                "offset": self.offset.cpu(),
                "depth_limit": self.depth_limit,
                "geom_resize_fact": self.geom_resize_fact,
                "data": self.data.data.half().cpu().numpy(),  # save CPU Memory
                "NodeCorners": self.NodeCorners.cpu(),
                "CornerSDF": self.CornerSDF.data.half().cpu().numpy(),
                "CornerSH": self.CornerSH.data.half().cpu().numpy(),
                "Beta": self.Beta.data.half().cpu().numpy(),
                "n_corner": self._n_corners.cpu().item(),
                "c_depth": self._c_depth.cpu().item()
            }
        if self.data_format is not None:
            data["data_format"] = repr(self.data_format)
        if self.extra_data is not None:
            data["extra_data"] = self.extra_data.cpu()
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

        np.save(path_d, self.CornerDict)

    def LOTSaveR(self, path, compress=False):
        beta = float(self.Beta.data.cpu().numpy()[0])
        data = {
            "data_dim": self.data_dim,
            "basis_dim": self.basis_dim,
            "child": self.child.cpu(),
            "parent_depth": self.parent_depth.cpu(),
            "n_internal": self._n_internal.cpu().item(),
            "n_free": self._n_free.cpu().item(),
            "invradius3": self.invradius.cpu(),
            "offset": self.offset.cpu(),
            "depth_limit": self.depth_limit,
            "geom_resize_fact": self.geom_resize_fact,
            "data": self.data.data.half().cpu().numpy(),  # save CPU Memory
            "NodeCorners": self.NodeCorners.cpu(),
            "CornerSDF": self.CornerSDF.data.half().cpu().numpy(),
            "CornerSH": self.CornerSH.data.half().cpu().numpy(),
            "Beta": beta
        }
        if self.data_format is not None:
            data["data_format"] = repr(self.data_format)
        if self.extra_data is not None:
            data["extra_data"] = self.extra_data.cpu()
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

    def LOTSaveFR(self, path, sample=False, compress=False):
        if sample:
            self.data.data = torch.zeros(self.child.size(0), 2, 2, 2,
                                         int(self.CornerSDF.size(1) + self.CornerSH.size(1)),
                                         dtype=torch.float32, device=self.CornerSH.device)

            sel = (*self._all_leaves().T,)
            leaf_nodes = torch.stack(sel, dim=-1).to(device=self.CornerSH.device)
            leaf_nodes = leaf_nodes.long()

            leaf_indices = self.LeafNodeMap[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3], 0]

            self.data.data[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3],
            0:self.CornerSH.size(1)] = self.CornerSH[leaf_indices.long(), :]

            self.data.data[leaf_nodes[:, 0], leaf_nodes[:, 1], leaf_nodes[:, 2], leaf_nodes[:, 3],
            self.CornerSH.size(1)] = self.CornerSDF[leaf_indices.long(), 0]

        data = {
            "data_dim": self.data_dim,
            "basis_dim": self.basis_dim,
            "child": self.child.cpu(),
            "parent_depth": self.parent_depth.cpu(),
            "n_internal": self._n_internal.cpu().item(),
            "n_free": self._n_free.cpu().item(),
            "invradius3": self.invradius.cpu(),
            "offset": self.offset.cpu(),
            "depth_limit": self.depth_limit,
            "geom_resize_fact": self.geom_resize_fact,
            "data": self.data.data.half().cpu().numpy(),  # save CPU Memory
            "Beta": self.Beta.data.cpu().numpy()
        }

        if self.data_format is not None:
            data["data_format"] = repr(self.data_format)
        if self.extra_data is not None:
            data["extra_data"] = self.extra_data.cpu()
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

    @classmethod
    def LOTLoadFR(cls, path, device='cpu', dtype=torch.float32):
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'
        tree = cls(dtype=dtype, device=device)
        z = np.load(path)
        tree.data_dim = int(z["data_dim"])
        tree.basis_dim = int(z["basis_dim"])
        tree.child = torch.from_numpy(z["child"]).to(device)
        tree.N = tree.child.shape[-1]
        tree.parent_depth = torch.from_numpy(z["parent_depth"]).to(device)
        tree.Beta.data = torch.from_numpy(z["Beta"].astype(np.float32)).to(device)
        tree._n_internal.fill_(z["n_internal"].item())
        if "invradius3" in z.files:
            tree.invradius = torch.from_numpy(z["invradius3"].astype(
                np.float32)).to(device)
        else:
            tree.invradius.fill_(z["invradius"].item())
        tree.offset = torch.from_numpy(z["offset"].astype(np.float32)).to(device)
        tree.depth_limit = int(z["depth_limit"])
        tree.geom_resize_fact = float(z["geom_resize_fact"])
        tree.data.data = torch.from_numpy(z["data"].astype(np.float32)).to(device)
        if 'n_free' in z.files:
            tree._n_free.fill_(z["n_free"].item())
        else:
            tree._n_free.zero_()
        tree.data_format = DataFormat(z['data_format'].item()) if \
            'data_format' in z.files else None
        tree.extra_data = torch.from_numpy(z['extra_data']).to(device) if \
            'extra_data' in z.files else None

        return tree

    @classmethod
    def LOTLoad(cls, path, path_d, load_full=True, load_dict=True, device='cpu', dtype=torch.float32):
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'
        tree = cls(dtype=dtype, device=device)
        z = np.load(path)
        tree.data_dim = int(z["data_dim"])
        tree.basis_dim = int(z["basis_dim"])
        tree.child = torch.from_numpy(z["child"]).to(device)
        tree.N = tree.child.shape[-1]
        tree.parent_depth = torch.from_numpy(z["parent_depth"]).to(device)
        tree._n_internal.fill_(z["n_internal"].item())
        if "invradius3" in z.files:
            tree.invradius = torch.from_numpy(z["invradius3"].astype(
                np.float32)).to(device)
        else:
            tree.invradius.fill_(z["invradius"].item())
        tree.offset = torch.from_numpy(z["offset"].astype(np.float32)).to(device)
        tree.depth_limit = int(z["depth_limit"])
        tree.geom_resize_fact = float(z["geom_resize_fact"])
        tree.data.data = torch.from_numpy(z["data"].astype(np.float32)).to(device)
        if 'n_free' in z.files:
            tree._n_free.fill_(z["n_free"].item())
        else:
            tree._n_free.zero_()
        tree.data_format = DataFormat(z['data_format'].item()) if \
            'data_format' in z.files else None
        tree.extra_data = torch.from_numpy(z['extra_data']).to(device) if \
            'extra_data' in z.files else None

        tree.CornerSDF.data = torch.from_numpy(z["CornerSDF"].astype(np.float32)).to(device)
        tree.CornerSH.data = torch.from_numpy(z["CornerSH"].astype(np.float32)).to(device)
        tree.Beta.data = torch.from_numpy(z["Beta"].astype(np.float32)).to(device)

        tree.NodeCorners = torch.from_numpy(z["NodeCorners"]).to(device)

        tree._n_corners.fill_(z["n_corner"].item())
        tree._c_depth.fill_(z["c_depth"].item())

        if load_full:
            tree.NodeAllNeighbors = torch.from_numpy(z["NodeAllNeighbors"]).to(device)
            tree.NodeAllNeighLen = torch.from_numpy(z["NodeAllNeighLen"].astype(np.float32)).to(device)

            tree.NodeGaussGradNeighbors = torch.from_numpy(z["NodeGaussGradNeighbors"]).to(device)
            tree.NodeGaussGradKernals = torch.from_numpy(z["NodeGaussGradKernals"].astype(np.float32)).to(device)

            tree.NodeGhoNeighbors = torch.from_numpy(z["NodeGhoNeighbors"]).to(device)
            tree.NodeGhoCoeff = torch.from_numpy(z["NodeGhoCoeff"].astype(np.float32)).to(device)

            tree.CornerMap = torch.from_numpy(z["CornerMap"]).to(device)

            tree.ValidGeoCorner = torch.from_numpy(z["ValidGeoCorner"]).to(device)
            tree.ValidGeoCornerCoord = torch.from_numpy(z["ValidGeoCornerCoord"]).to(device)\

        if load_dict:
            tree.CornerDict = np.load(path_d, allow_pickle=True).item()

        return tree

    @property
    def n_corners(self):
        return self._n_corners.item()

    @property
    def c_depth(self):
        return self._c_depth.item()

    def __getitem__(self, key):
        return LONodeA(self, key)

    def __setitem__(self, key, val):
        N3TreeView(self, key).set(val)


def CreateNewTreeBasedPast(past_tree, refine_time, n_samples,
                           sdf_thresh, geo_sdf_thresh, sdf_thresh_offset, dilate_times,
                           device):
    center, radius, bbox_corner, bbox_length = past_tree.BoundOctree(n_samples=n_samples,
                                                                     sdf_thresh=sdf_thresh,
                                                                     sdf_offset=sdf_thresh_offset,
                                                                     bbox_scale=1.3)

    bbox_center = center.detach().cpu()
    bbox_radius = radius.detach().cpu()

    new_tree = LOctreeA(N=2,
                        data_dim=28,
                        basis_dim=9,
                        init_refine=0,
                        init_reserve=50000,
                        geom_resize_fact=1.0,
                        center=bbox_center,
                        radius=bbox_radius,
                        depth_limit=11,
                        data_format=f'SH{9}',
                        device=device)
    new_tree.InitializeCorners()

    for i in range(5):
        new_tree.RefineCorners()

    corners = torch.arange(0, new_tree.CornerSDF.size(0), dtype=torch.int32, device=device)
    new_tree.SetGeoCorner(corner=corners)
    corners = new_tree.ValidGeoCornerCoord / (2 ** (new_tree.depth_limit + 1))
    new_tree_sdf, new_tree_sh = past_tree.ComDataForAnoTree(corners, bbox_length, bbox_corner, com_sh=True)

    new_tree.CornerSDF.data[new_tree.ValidGeoCorner.long()] = new_tree_sdf
    new_tree.CornerSH.data[new_tree.ValidGeoCorner.long()] = new_tree_sh

    for i in range(refine_time):
        sel = (*new_tree._all_leaves().T,)
        leaf_nodes = torch.stack(sel, dim=-1).to(device=device)

        l_node_corners = new_tree.CalCorner(leaf_nodes)
        node_depths = new_tree.parent_depth[leaf_nodes[:, 0].long(), 1]
        node_lengths = 1.0 / (2 ** (node_depths + 1))

        offsets = torch.rand((l_node_corners.shape[0], n_samples, 3), device=device, dtype=torch.float32)
        offsets = node_lengths[:, None, None] * offsets

        sample_points = l_node_corners[:, None] + offsets
        sample_points = sample_points.view(-1, 3)
        new_tree_sdf, _ = past_tree.ComDataForAnoTree(sample_points, bbox_length, bbox_corner)

        # new_tree_sdf = new_tree_sdf.view(-1)
        # inside_sdf_mask = (-new_tree_sdf > sdf_thresh)
        # new_tree_sdf[inside_sdf_mask] += sdf_thresh_offset
        new_tree_sdf = new_tree_sdf.view(-1, n_samples)

        # new_tree_sdf = new_tree_sdf.abs()
        new_tree_sdf = torch.min(new_tree_sdf, dim=1)[0]

        sdf_mask = (new_tree_sdf <= geo_sdf_thresh)
        select_leaf_nodes = leaf_nodes[sdf_mask]

        select_leaf_nodes = new_tree.DilateCorners(d_nodes=select_leaf_nodes, dilate_times=dilate_times)

        sel = (*select_leaf_nodes.T,)
        new_corners, _ = new_tree.RefineCorners(sel=sel, tri_inside=False)

        new_corners = new_corners.view(-1, 3)
        new_corners = torch.unique(new_corners, dim=0)
        corner_indices = new_tree.FastFromPosGetCornerIndex(new_corners)
        new_corners = new_corners / (2 ** (new_tree.depth_limit + 1))
        new_corners = new_corners.to(device)
        new_tree_sdf, new_tree_sh = past_tree.ComDataForAnoTree(new_corners, bbox_length, bbox_corner, com_sh=True)
        new_tree.CornerSDF.data[corner_indices.long()] = new_tree_sdf
        new_tree.CornerSH.data[corner_indices.long()] = new_tree_sh

    # new_tree.data.data = torch.zeros(new_tree.child.size(0), 2, 2, 2, 28, dtype=torch.float32, device=device)
    # new_tree.LOTSaveN(path='D:/LOTree/LOTree/apple/vischair.npz')

    sel = (*new_tree._all_leaves().T,)
    leaf_nodes = torch.stack(sel, dim=-1).to(device=device)
    new_tree.FilterGeoCorners(leaf_nodes, geo_sdf_thresh, sdf_thresh_offset, False, n_samples,
                              dilate_times=dilate_times)

    new_tree.FindAllNeighCorner()
    new_tree.FindGhostNeighbors()

    # ksize = 5
    # new_tree.NodeGaussNeighbors, new_tree.NodeGaussKernals = new_tree.FindGaussianNeigh(ksize=ksize, sigma=0.8)

    ksize = 3
    new_tree.NodeGaussGradNeighbors, new_tree.NodeGaussGradKernals = new_tree.FindCubicGaussianNeigh(sigma=0.8)
    new_tree.NodeGaussGradKernals[:, int(ksize * ksize * ksize / 2)] = new_tree.NodeGaussGradKernals[:,
                                                                       int(ksize * ksize * ksize / 2)] - 1.0

    return new_tree, bbox_corner, bbox_length
