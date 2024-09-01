#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""Optimize a plenoctree through finetuning on train set.

Usage:

export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/PlenOctree/checkpoints/syn_sh16
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.optimization \
    --input $CKPT_ROOT/$SCENE/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree_opt.npz
"""
import gc

import svox
import torch
import torch.cuda

import numpy as np
import random
import json
import imageio
import os.path as osp
import os
import scipy.io as io
from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn as nn
from torch.optim import SGD, Adam
from warnings import warn

from absl import app
from absl import flags

from svox import utils
from svox2 import defs
from svox import datasets
from LOTRenderer import LOTRenderA
from LOTDataset.Dataset import datasetsLOT

from LOctree import LOctreeA, CreateNewTreeBasedPast
from LOTCorr import RenderOptions, Rays, Camera
from svox.helpers import LocalIndex

from timeit import default_timer as timer

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string("data_dir1",
                    "D:/LOTree/LOTree/apple",
                    "input data directory."
                    )
flags.DEFINE_string(
    "input",
    "D:/LOTree/LOTree/apple/tree.npz",
    "Input octree npz from extraction.py",
)
flags.DEFINE_string(
    "output",
    "D:/LOTree/LOTree/apple/tree_opt.npz",
    "Output octree npz",
)
flags.DEFINE_integer(
    'render_interval',
    0,
    'render interval')
flags.DEFINE_integer(
    'val_interval',
    2,
    'validation interval')
flags.DEFINE_integer(
    'test_every',
    1,
    'epochs to test for')
flags.DEFINE_bool(
    'sgd',
    True,
    'use SGD optimizer instead of Adam')

flags.DEFINE_float(
    'lr',
    1e7,
    'optimizer step size')

flags.DEFINE_float(
    'lr_sigma',
    3e1,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sigma_final',
    5e-2,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sigma_decay_steps',
    250000,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sigma_delay_steps',
    15000,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sigma_delay_mult',
    1e-2,
    'optimizer step size')

flags.DEFINE_float(
    'lr_sdf',
    1e-5,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sdf_final',
    1e-5,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sdf_decay_steps',
    51200,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sdf_delay_steps',
    0,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sdf_delay_mult',
    1e-2,
    'optimizer step size')

flags.DEFINE_float(
    'lr_beta',
    1e-2,
    'optimizer step size')
flags.DEFINE_float(
    'lr_beta_final',
    1e-5,
    'optimizer step size')
flags.DEFINE_float(
    'lr_beta_decay_steps',
    250000,
    'optimizer step size')
flags.DEFINE_float(
    'lr_beta_delay_steps',
    15000,
    'optimizer step size')
flags.DEFINE_float(
    'lr_beta_delay_mult',
    1e-2,
    'optimizer step size')

flags.DEFINE_float(
    'lr_learns',
    5e-4,
    'optimizer step size')
flags.DEFINE_float(
    'lr_learns_final',
    5e-4,
    'optimizer step size')
flags.DEFINE_float(
    'lr_learns_decay_steps',
    250000,
    'optimizer step size')
flags.DEFINE_float(
    'lr_learns_delay_steps',
    15000,
    'optimizer step size')
flags.DEFINE_float(
    'lr_learns_delay_mult',
    1e-2,
    'optimizer step size')

flags.DEFINE_float(
    'lr_sh',
    1e-3,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sh_final',
    1e-3,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sh_decay_steps',
    250000,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sh_delay_steps',
    0,
    'optimizer step size')
flags.DEFINE_float(
    'lr_sh_delay_mult',
    1e-2,
    'optimizer step size')

flags.DEFINE_float(
    'lambda_tv',
    1e-5,
    'optimizer step size')
flags.DEFINE_float(
    'tv_sparsity',
    1e-2,
    'optimizer step size')
flags.DEFINE_float(
    'lambda_tv_sh',
    1e-3,
    'optimizer step size')

flags.DEFINE_float(
    'rms_beta',
    0.95,
    'optimizer step size')

flags.DEFINE_float(
    'sgd_momentum',
    0.9,
    'sgd momentum')
flags.DEFINE_bool(
    'sgd_nesterov',
    False,
    'sgd nesterov momentum?')
flags.DEFINE_string(
    "write_vid",
    None,
    "If specified, writes rendered video to given path (*.mp4)",
)

# Manual 'val' set
flags.DEFINE_bool(
    "split_train",
    True,
    "If specified, splits train set instead of loading val set",
)
flags.DEFINE_float(
    "split_holdout_prop",
    0.2,
    "Proportion of images to hold out if split_train is set",
)

# Do not save since it is slow
flags.DEFINE_bool(
    "nosave",
    False,
    "If set, does not save (for speed)",
)

flags.DEFINE_bool(
    "continue_on_decrease",
    False,
    "If set, continues training even if validation PSNR decreases",
)

device = "cuda"  # "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def main(unused_argv):
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)

    print('LOTreeA load')
    vis_dir = osp.splitext(FLAGS.input)[0] + '_render'

    # t = LOctreeA.LOTLoad(path="D:/LOTree/LOTree/apple/LOT_new_1121chair_1024_full.npz",
    #                      path_d="D:/LOTree/LOTree/apple/cor_dict_new_1121chair_1024_full.npy",
    #                      device=device,
    #                      dtype=torch.float32)
    # t = LOctreeA.LOTLoadN(path="D:/LOTree/LOTree/VolsdfSH/LOTDTU_122_2.npz",
    #                       device=device,
    #                       dtype=torch.float32)
    # t = LOctreeA.LOTLoad(path="D:/LOTree/LOTree/VolsdfSH/LOTDTU_106_1.npz",
    #                      path_d='D:/LOTree/LOTree/VolsdfSH/LOTDTU_106_1_dic.npy',
    #                      load_full=False,
    #                      device=device,
    #                      dtype=torch.float32)

    t.Beta.data[0] = 0.0002
    # t.CornerSH.data = torch.zeros(1, 27, dtype=torch.float32, device=device)

    t.density_rms = None
    t.sh_rms = None
    t.sdf_rms = None
    t.beta_rms = None
    t.data_sh_rms = None
    t.data_sdf_rms = None
    t.opt = RenderOptions()
    t.basis_type = defs.BASIS_TYPE_SH
    t.offset = t.offset.to('cpu')
    t.invradius = t.invradius.to('cpu')

    t._invalidate()

    # t.LOTSaveN(path="D:/LOTree/LOTree/apple/LOTChairN.npz")

    dset_train = datasetsLOT[FLAGS.dataset](
        FLAGS.data_dir1,
        split="train",
        device=device,
        factor=1.0,
        n_images=90,
        **utils.build_data_options(FLAGS))

    dset_test = datasetsLOT[FLAGS.dataset](
        FLAGS.data_dir1,
        split="test",
        factor=1.0,
        n_images=64,
        **utils.build_data_options(FLAGS))

    os.makedirs(vis_dir, exist_ok=True)

    lr_sh_func = get_expon_lr_func(FLAGS.lr_sh, FLAGS.lr_sh_final, FLAGS.lr_sh_delay_steps,
                                   FLAGS.lr_sh_delay_mult, FLAGS.lr_sh_decay_steps)
    lr_sdf_func = get_expon_lr_func(FLAGS.lr_sdf, FLAGS.lr_sdf_final, FLAGS.lr_sdf_delay_steps,
                                    FLAGS.lr_sdf_delay_mult, FLAGS.lr_sdf_decay_steps)

    epoch_id = -1
    gstep_id_base = 0

    radius = 0.5 / t.invradius
    center = (1 - 2.0 * t.offset) * radius
    bbox_corner = center - radius
    bbox_corner.cuda()
    bbox_length = radius * 2
    bbox_length.cuda()

    t.opt.cube_thresh = 512
    t.stop_thresh = 1e-3
    # sel = (*t._all_leaves().T,)
    # leaf_nodes = torch.stack(sel, dim=-1).to(device=device)
    #
    # t.FilterGeoCorners(leaf_nodes, 0.005, 0.001, True, 256, 1)
    # t.FindAllNeighCorner()
    # t.FindGhostNeighbors()
    #
    # ksize = 3
    # t.NodeGaussGradNeighbors, t.NodeGaussGradKernals = t.FindCubicGaussianNeigh(sigma=0.8)
    # t.NodeGaussGradKernals[:, int(ksize * ksize * ksize / 2)] = t.NodeGaussGradKernals[:,
    #                                                             int(ksize * ksize * ksize / 2)] - 1.0

    # leaf_nodes = leaf_nodes.long()

    while True:
        epoch_id += 1

        dset_train.shuffle_rays()

        epoch_size = dset_train.rays.origins.size(0)
        batches_per_epoch = (epoch_size - 1) // FLAGS.batch_size + 1

        def train_step():
            print('Train step')

            tpsnr = 0.0
            pbar = tqdm(enumerate(range(0, epoch_size, FLAGS.batch_size)), total=batches_per_epoch)
            for iter_id, batch_begin in pbar:
                gstep_id = iter_id + gstep_id_base

                lr_sh = lr_sh_func(gstep_id)
                lr_sdf = lr_sdf_func(gstep_id)

                batch_end = min(batch_begin + FLAGS.batch_size, epoch_size)
                batch_origins = dset_train.rays.origins[batch_begin: batch_end]
                batch_dirs = dset_train.rays.dirs[batch_begin: batch_end]

                im_gt = dset_train.rays.gt[batch_begin: batch_end]

                rays = Rays(batch_origins, batch_dirs)

                im = t.VolumeRenderConvert(rays, im_gt, scale=5e-4, ksize_grad=3,
                                           sparse_frac=0.1,
                                           gauss_grad_loss=False)

                mse = ((im - im_gt) ** 2).mean()

                psnr = -10.0 * np.log(mse.detach().cpu()) / np.log(10.0)

                print('step: ', gstep_id)
                print('psnr: ', psnr)
                tpsnr += psnr.item()

                # t.OptimData(lr_sh=lr_sh, lr_sdf=lr_sdf, beta=FLAGS.rms_beta, leaf_nodes=leaf_nodes)
                t.OptimSDF(lr_sdf, beta=FLAGS.rms_beta)
                t.OptimSH(lr_sh, beta=FLAGS.rms_beta)

                # t.Beta.data[0] = 1.0 / (gstep_id / 6.0 + 5000.0)

                if gstep_id % 5000 == 0 and gstep_id != 0:
                    test_step()
                    gc.collect()

                # if gstep_id % 10000 == 0:
                #     mesh_dir = "D:/LOTree/LOTree/apple"
                #
                #     if epoch_id >= 4:
                #         resolution = 1024
                #     else:
                #         resolution = 512
                #
                #     t.ExtractGeometry(resolution, bbox_corner, bbox_length, mesh_dir, iter_step=gstep_id)
                #     gc.collect()

            print('** train_psnr', tpsnr)

        def test_step():
            with torch.no_grad():
                tree_rad = 0.5 / t.invradius
                t.RenderSDFIsolines(z_pos=0.5, radius=tree_rad, vis_dir=vis_dir, epoch_id=epoch_id)

                # t.offset = t.offset.to(device)
                # t.invradius = t.invradius.to(device)

                N_IMGS_TO_EVAL = 10
                img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
                img_ids = range(0, dset_test.n_images, img_eval_interval)

                total_psnr = 0.0

                for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                    c2w = dset_test.c2w[img_id].to(device=device)

                    width = dset_test.get_image_size(img_id)[1]
                    height = dset_test.get_image_size(img_id)[0]

                    rays = LOTRenderA.GenerateRaysByCamera(c2w,
                                                           dset_test.intrins.get('fx', img_id),
                                                           dset_test.intrins.get('fy', img_id),
                                                           dset_test.intrins.get('cx', img_id),
                                                           dset_test.intrins.get('cy', img_id),
                                                           width, height)

                    rays.origins.view(width * height, -1)
                    rays.dirs.view(width * height, -1)

                    rays_r = Rays(rays.origins, rays.dirs)

                    im_gt = dset_test.gt[img_id].to(device='cpu')
                    im_gt_ten = im_gt.view(width * height, -1)

                    im = t.VolumeRenderVolSDFCVTTest(rays_r)

                    im = im.cpu().clamp_(0.0, 1.0)

                    im = im.view(height, width, -1)
                    im = im * dset_test.masks[img_id] + (1 - dset_test.masks[img_id])

                    im = im.view(width * height, -1)

                    mse = ((im - im_gt_ten) ** 2).mean()
                    psnr = -10.0 * np.log(mse) / np.log(10.0)

                    total_psnr += psnr.item()

                    print('psnr: ', psnr)

                    # im = im.cpu().clamp_(0.0, 1.0)

                    # mse = ((im - im_gt_ten) ** 2).mean()
                    # psnr = -10.0 * np.log(mse) / np.log(10.0)
                    # print('psnr: ', psnr)

                    im = im.view(height, width, -1)
                    vis = torch.cat((im_gt, im), dim=1)
                    vis = (vis * 255).numpy().astype(np.uint8)
                    imageio.imwrite(f"{vis_dir}/{i:04}_{img_id:04}.png", vis)

                total_psnr /= N_IMGS_TO_EVAL
                print('Avg PSNR: ', total_psnr)

        if epoch_id == 1 or epoch_id == 2 or epoch_id == 4 or epoch_id == 6 or epoch_id == 8:
            if epoch_id == 6:
                t.LOTSave(path='D:/LOTree/LOTree/VolsdfSH/LOTDTU_106_trifine.npz',
                          path_d='D:/LOTree/LOTree/VolsdfSH/LOTDTU_106_trifine_dic.npy', out_full=False)

                gc.collect()

        # if epoch_id == 3:
        #     t.VolumeRenderVolSDFRecord(dset_train=dset_train)

        # test_step()
        train_step()
        print(torch.cuda.memory_allocated())
        print(torch.cuda.max_memory_allocated())
        gc.collect()
        gstep_id_base += 12800


if __name__ == "__main__":
    app.run(main)
