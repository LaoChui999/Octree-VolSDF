import gc

import torch

from pyhocon import ConfigFactory
from tqdm import tqdm
from absl import app
from VolsdfSH.model.network import VolSDFNetwork
from LOctree import LOctreeA
from svox2 import utils
from LOTCorr import Rays

device = "cuda"
_C = utils._get_c_extension()


def ComBBoxForOctree(reso, volsdf_sh, sdf_thresh):
    radius = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    scale = 0.5 / radius
    offset = 0.5 * (1.0 - center / radius)

    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]

    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T

    chunk_size = 20000

    all_sample_sdf = []
    for i in tqdm(range(0, grid.size(0), chunk_size)):
        grid_chunk = grid[i:i + chunk_size].cuda()

        sdf = volsdf_sh.implicit_network.GetSDF(grid_chunk)
        del grid_chunk

        sdf = sdf.cpu()

        all_sample_sdf.append(sdf)

    all_sample_sdf = torch.cat(all_sample_sdf, dim=0)

    sdf_mask = (all_sample_sdf <= sdf_thresh)
    grid = grid[sdf_mask]

    lc = grid.min(dim=0)[0] - 0.5 / reso
    uc = grid.max(dim=0)[0] + 0.5 / reso

    center = (lc + uc) * 0.5
    radius = (uc - lc) * 0.5

    return center, radius


def CreateOctreeBasedVolSDF(center, radius, volsdf_sh, sdf_thresh, refine_time, n_samples):
    new_tree = LOctreeA(N=2,
                        data_dim=28,
                        basis_dim=9,
                        init_refine=0,
                        init_reserve=50000,
                        geom_resize_fact=1.0,
                        center=center,
                        radius=radius,
                        depth_limit=11,
                        data_format=f'SH{9}',
                        device=device)
    new_tree.InitializeCorners()

    for i in range(5):
        new_tree.RefineCorners()

    c_sel = (*new_tree._all_leaves().T,)
    for i in range(refine_time):
        c_leaf_nodes = torch.stack(c_sel, dim=-1).to(device=device)

        chunk_size = 10000
        all_refine_nodes = []
        for j in tqdm(range(0, c_leaf_nodes.size(0), chunk_size)):
            cc_leaf_nodes = c_leaf_nodes[j:j + chunk_size]

            torch.cuda.empty_cache()

            l_node_corners = new_tree.CalCorner(cc_leaf_nodes)
            node_depths = new_tree.parent_depth[cc_leaf_nodes[:, 0].long(), 1]
            node_lengths = 1.0 / (2 ** (node_depths + 1))

            offsets = torch.rand((l_node_corners.shape[0], n_samples, 3), device=device, dtype=torch.float32)
            offsets = node_lengths[:, None, None] * offsets

            sample_points = l_node_corners[:, None] + offsets
            sample_points = sample_points.view(-1, 3)

            sample_points[:, 0] = (sample_points[:, 0] - new_tree.offset[0]) / new_tree.invradius[0]
            sample_points[:, 1] = (sample_points[:, 1] - new_tree.offset[1]) / new_tree.invradius[1]
            sample_points[:, 2] = (sample_points[:, 2] - new_tree.offset[2]) / new_tree.invradius[2]

            sdf = volsdf_sh.implicit_network.GetSDF(sample_points)
            sdf = sdf.view(-1, n_samples)

            if i > 0:
                sdf = sdf.abs()

            sdf = sdf.min(dim=1)[0]

            sdf_mask = (sdf <= sdf_thresh)

            cc_leaf_nodes = cc_leaf_nodes[sdf_mask]
            cc_leaf_nodes = cc_leaf_nodes.cpu()
            all_refine_nodes.append(cc_leaf_nodes)
        all_refine_nodes = torch.cat(all_refine_nodes, dim=0)
        c_sel = (*all_refine_nodes.T,)

        _, c_sel = new_tree.RefineCorners(sel=c_sel)

    # new_tree.LOTSaveN(path='D:/LOTree/LOTree/VolsdfSH/LOTDTU_122_0.npz')
    #
    # new_tree = LOctreeA.LOTLoadN(path="D:/LOTree/LOTree/VolsdfSH/LOTDTU_122_0.npz",
    #                              device=device,
    #                              dtype=torch.float32)
    # new_tree._invalidate()

    c_sel = (*new_tree._all_leaves().T,)
    c_leaf_nodes = torch.stack(c_sel, dim=-1).to(device=device)

    chunk_size = 2000
    all_nodes_sdf = []
    all_nodes_sh = []
    for i in tqdm(range(0, c_leaf_nodes.size(0), chunk_size)):
        torch.cuda.empty_cache()

        l_node_corners = new_tree.CalCorner(c_leaf_nodes[i:i + chunk_size])
        node_depths = new_tree.parent_depth[c_leaf_nodes[i:i + chunk_size, 0].long(), 1]
        node_lengths = 1.0 / (2 ** (node_depths + 1))

        offsets = torch.rand((l_node_corners.shape[0], n_samples, 3), device=device, dtype=torch.float32)
        offsets = node_lengths[:, None, None] * offsets

        c_sample_points = l_node_corners[:, None] + offsets
        c_sample_points = c_sample_points.view(-1, 3)

        c_sample_points[:, 0] = (c_sample_points[:, 0] - new_tree.offset[0]) / new_tree.invradius[0]
        c_sample_points[:, 1] = (c_sample_points[:, 1] - new_tree.offset[1]) / new_tree.invradius[1]
        c_sample_points[:, 2] = (c_sample_points[:, 2] - new_tree.offset[2]) / new_tree.invradius[2]

        sdf, feature_vectors, gradients = volsdf_sh.implicit_network.get_outputs(c_sample_points)

        with torch.no_grad():
            sdf = sdf.view(-1, n_samples)
            sdf = sdf.mean(dim=1)
            sdf = sdf.cpu()
            all_nodes_sdf.append(sdf)

            fake_viewdirs = torch.zeros([c_sample_points.size(0), 3], device=c_sample_points.device)
            sh = volsdf_sh.rendering_network.GetSHFromPoints(c_sample_points, gradients, fake_viewdirs, feature_vectors)

            sh = sh.view(-1, n_samples, 27)
            sh = sh.mean(dim=1)
            sh = sh.cpu()
            all_nodes_sh.append(sh)

    all_nodes_sdf = torch.cat(all_nodes_sdf, dim=0)
    all_nodes_sh = torch.cat(all_nodes_sh, dim=0)
    all_nodes_sdf = all_nodes_sdf.cuda()
    all_nodes_sh = all_nodes_sh.cuda()

    c_leaf_nodes.long()
    new_tree.data.data = torch.zeros(new_tree.child.size(0), 2, 2, 2, 28,
                                     dtype=torch.float32, device=device)
    c_leaf_nodes = c_leaf_nodes.long()
    new_tree.data.data[c_leaf_nodes[:, 0], c_leaf_nodes[:, 1], c_leaf_nodes[:, 2], c_leaf_nodes[:, 3],
    0:all_nodes_sh.size(1)] = all_nodes_sh
    new_tree.data.data[c_leaf_nodes[:, 0], c_leaf_nodes[:, 1], c_leaf_nodes[:, 2], c_leaf_nodes[:, 3],
    all_nodes_sh.size(1)] = all_nodes_sdf

    new_tree.LOTSaveN(path='D:/LOTree/LOTree/VolsdfSH/LOTDTU_122_2.npz')

    # new_tree.CornerSDF.data[new_tree.ValidGeoCorner.long(), 0] = all_corners_sdf
    #
    # tree_rad = 0.5 / new_tree.invradius
    # vis_dir = 'D:/LOTree/LOTree/VolsdfSH'
    # new_tree.RenderSDFIsolines(z_pos=0.5, radius=tree_rad, vis_dir=vis_dir, epoch_id=0)
    #
    # bbox_corner = center - radius
    # bbox_corner.cuda()
    # bbox_length = radius * 2
    # bbox_length.cuda()
    # new_tree.ExtractGeometry(512, bbox_corner, bbox_length, vis_dir, iter_step=0)


def CreateOctreeBasedVolSDFTri(center, radius, volsdf_sh, sdf_thresh, refine_time, model_id):
    new_tree = LOctreeA(N=2,
                        data_dim=28,
                        basis_dim=9,
                        init_refine=0,
                        init_reserve=50000,
                        geom_resize_fact=1.0,
                        center=center,
                        radius=radius,
                        depth_limit=11,
                        data_format=f'SH{9}',
                        device=device)
    new_tree.InitializeCorners()

    for i in range(5):
        new_tree.RefineCorners()

    sample_indices = torch.arange(0, new_tree.CornerSDF.size(0), dtype=torch.int32, device=device)
    inverse_dict = {v: k for k, v in new_tree.CornerDict.items()}
    sample_points = new_tree.FromCornerIndexGetPos(sample_indices, inverse_dict)
    sample_points = sample_points / (2 ** (new_tree.depth_limit + 1))
    sample_indices = sample_indices.long()

    for i in range(refine_time + 1):
        chunk_size = int(1e6)
        all_sdf = []
        for j in tqdm(range(0, sample_points.size(0), chunk_size)):
            c_sample_points = sample_points[j:j + chunk_size]

            c_sample_points[:, 0] = (c_sample_points[:, 0] - new_tree.offset[0]
                                     ) / new_tree.invradius[0]
            c_sample_points[:, 1] = (c_sample_points[:, 1] - new_tree.offset[1]
                                     ) / new_tree.invradius[1]
            c_sample_points[:, 2] = (c_sample_points[:, 2] - new_tree.offset[2]
                                     ) / new_tree.invradius[2]

            sdf = volsdf_sh.implicit_network.GetSDF(c_sample_points)
            all_sdf.append(sdf)

            del c_sample_points

        all_sdf = torch.cat(all_sdf, dim=0)
        new_tree.CornerSDF.data[sample_indices, 0] = all_sdf

        tree_rad = 0.5 / new_tree.invradius
        vis_dir = 'D:/LOTree/LOTree/VolsdfSH'
        new_tree.RenderSDFIsolines(z_pos=0.5, radius=tree_rad, vis_dir=vis_dir, epoch_id=0)

        bbox_corner = center - radius
        bbox_corner.cuda()
        bbox_length = radius * 2
        bbox_length.cuda()
        new_tree.ExtractGeometry(512, bbox_corner, bbox_length, vis_dir, iter_step=0)

        if i < refine_time:
            c_sel = (*new_tree._all_leaves().T,)
            c_leaf_nodes = torch.stack(c_sel, dim=-1).to(device=new_tree.CornerSH.device)

            if i == 0:
                is_abs = False
            else:
                is_abs = True

            c_sdf = new_tree.SampleSDF(nodes=c_leaf_nodes, n_samples=256, sdf_thresh=sdf_thresh, sdf_offset=0,
                                       is_abs=is_abs)

            sdf_mask = (c_sdf <= sdf_thresh)
            select_leaf_nodes = c_leaf_nodes[sdf_mask]
            select_leaf_nodes = new_tree.DilateCorners(d_nodes=select_leaf_nodes, dilate_times=1)

            sel = (*select_leaf_nodes.T,)
            sample_points, _ = new_tree.RefineCorners(sel=sel, tri_inside=False)

            sample_points = sample_points.view(-1, 3)
            sample_points = torch.unique(sample_points, dim=0)

            sample_indices = new_tree.FastFromPosGetCornerIndex(sample_points)
            sample_indices = sample_indices.long()

            sample_points = sample_points / (2 ** (new_tree.depth_limit + 1))
            sample_points = sample_points.to(device)

    sample_indices = torch.arange(0, new_tree.CornerSDF.size(0), dtype=torch.int32, device=device)
    inverse_dict = {v: k for k, v in new_tree.CornerDict.items()}
    sample_points = new_tree.FromCornerIndexGetPos(sample_indices, inverse_dict)
    sample_points = sample_points / (2 ** (new_tree.depth_limit + 1))
    sample_indices = sample_indices.long()

    all_sh = GetSHFromVolSH(sample_points, volsdf_sh, new_tree.offset, new_tree.invradius)
    new_tree.CornerSH.data[sample_indices] = all_sh

    path = 'D:/LOTree/LOTree/VolsdfSH/LOTDTU_' + str(model_id) + '.npz'
    pathd = 'D:/LOTree/LOTree/VolsdfSH/LOTDTU_' + str(model_id) + '_dic.npy'

    new_tree.LOTSave(path=path,
                     path_d=pathd, out_full=False)

    return new_tree


def AdaptiveSubdivideColor(
        tree,
        dset_train,
        sdf_thresh: float = 0.0,
        max_corner_threshold: int = 3e8,
        sdf_threshold: float = 0.0):
    chunk_size = 10000
    node_mse = -1 * 999 * torch.ones(tree.child.size(0), tree.child.size(1), tree.child.size(2), tree.child.size(3),
                                     1, dtype=torch.float32, device=tree.CornerSH.device)

    for i in tqdm(range(0, dset_train.rays.origins.size(0), chunk_size)):
        chunk_origins = dset_train.rays.origins[i: i + chunk_size]
        chunk_dirs = dset_train.rays.dirs[i: i + chunk_size]

        im_gt = dset_train.rays.gt[i: i + chunk_size]

        rays = Rays(chunk_origins, chunk_dirs)

        rgb_out = torch.zeros(rays.origins.size(0), 3, dtype=torch.float32, device=tree.CornerSH.device)

        cuda_tree = tree._LOTspecA()
        cuda_opt = tree.opt._to_cpp()

        cu_fn = _C.__dict__[f"volume_render_volsdf_test_LOT"]

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

    sel = (*tree._all_leaves().T,)
    leaf_nodes = torch.stack(sel, dim=-1).to(device=tree.CornerSH.device)

    leaf_sdf = tree.SampleSDF(nodes=leaf_nodes, n_samples=256, sdf_thresh=sdf_thresh, sdf_offset=0,
                              is_abs=True)

    node_mask = (leaf_sdf <= sdf_threshold)
    select_leaf_nodes = leaf_nodes[node_mask]

    select_leaf_nodes = tree.DilateCorners(d_nodes=select_leaf_nodes, dilate_times=1)

    select_node_mse = node_mse[select_leaf_nodes[:, 0].long(), select_leaf_nodes[:, 1].long(),
    select_leaf_nodes[:, 2].long(), select_leaf_nodes[:, 3].long()]

    select_node_mse, sort_index = torch.sort(select_node_mse, dim=0, descending=True)

    capacity_free = float(max_corner_threshold - tree.CornerSH.size(0))

    if capacity_free > 0:
        node_refine_num = int(capacity_free / 7.0)

        c_node_mse = select_node_mse[node_refine_num - 1]
        for i in range(node_refine_num, select_node_mse.size(0), 1):
            if select_node_mse[i] != c_node_mse:
                node_refine_num = i
                break

        sort_index = sort_index.view(-1)
        select_leaf_nodes = select_leaf_nodes[sort_index]
        select_leaf_nodes = select_leaf_nodes[:node_refine_num]

        select_leaf_nodes = tree.DilateCorners(d_nodes=select_leaf_nodes, dilate_times=1)


def GetSHFromVolSH(sample_points, volsdf_sh, offset, invradius):
    chunk_size = int(1e5)
    all_sh = []
    for i in tqdm(range(0, sample_points.size(0), chunk_size)):
        c_sample_points = sample_points[i:i + chunk_size]

        c_sample_points[:, 0] = (c_sample_points[:, 0] - offset[0]
                                 ) / invradius[0]
        c_sample_points[:, 1] = (c_sample_points[:, 1] - offset[1]
                                 ) / invradius[1]
        c_sample_points[:, 2] = (c_sample_points[:, 2] - offset[2]
                                 ) / invradius[2]

        _, feature_vectors, gradients = volsdf_sh.implicit_network.get_outputs(c_sample_points)

        with torch.no_grad():
            fake_viewdirs = torch.zeros([c_sample_points.size(0), 3], device=c_sample_points.device)
            sh = volsdf_sh.rendering_network.GetSHFromPoints(c_sample_points, gradients, fake_viewdirs, feature_vectors)
            all_sh.append(sh)
    all_sh = torch.cat(all_sh, dim=0)
    all_sh = all_sh.cuda()

    return all_sh


def PatchConversion(conf_file_path, check_point_path, model_id):
    conf_file_path = conf_file_path
    check_point_path = check_point_path

    volsdf_conf = ConfigFactory.parse_file(conf_file_path)

    conf_model = volsdf_conf.get_config('model')
    volsdf_sh_model = VolSDFNetwork(conf_model)
    volsdf_sh_model.cuda()

    saved_model_state = torch.load(check_point_path)
    volsdf_sh_model.load_state_dict(saved_model_state["model_state_dict"])

    center, radius = ComBBoxForOctree(256, volsdf_sh_model, 0.3)

    CreateOctreeBasedVolSDFTri(center, radius, volsdf_sh_model, 0.008, 4, model_id)


def main(unused_argv):
    conf_file_path = 'D:/LOTree/LOTree/VolsdfSH/confs/dtu.conf'
    check_point_path = 'D:/LOTree/LOTree/VolsdfSH/latest_dtu106_1.pth'

    volsdf_conf = ConfigFactory.parse_file(conf_file_path)

    conf_model = volsdf_conf.get_config('model')
    volsdf_sh_model = VolSDFNetwork(conf_model)
    volsdf_sh_model.cuda()

    saved_model_state = torch.load(check_point_path)
    volsdf_sh_model.load_state_dict(saved_model_state["model_state_dict"])

    center, radius = ComBBoxForOctree(256, volsdf_sh_model, 0.3)

    # CreateOctreeBasedVolSDF(center, radius, volsdf_sh_model, 0.05, 3, 128)
    CreateOctreeBasedVolSDFTri(center, radius, volsdf_sh_model, 0.008, 4)


if __name__ == "__main__":
    app.run(main)
