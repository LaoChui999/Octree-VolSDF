// Copyright 2021 Alex Yu
#pragma once
#include <torch/extension.h>
#include "data_spec.hpp"
#include "cuda_util.cuh"
#include "random_util.cuh"

namespace
{
    namespace device
    {

        struct PackedSparseGridSpec
        {
            PackedSparseGridSpec(SparseGridSpec &spec)
                : density_data(spec.density_data.data_ptr<float>()),
                  sh_data(spec.sh_data.data_ptr<float>()),
                  links(spec.links.data_ptr<int32_t>()),
                  basis_type(spec.basis_type),
                  basis_data(spec.basis_data.defined() ? spec.basis_data.data_ptr<float>() : nullptr),
                  background_links(spec.background_links.defined() ? spec.background_links.data_ptr<int32_t>() : nullptr),
                  background_data(spec.background_data.defined() ? spec.background_data.data_ptr<float>() : nullptr),
                  size{(int)spec.links.size(0),
                       (int)spec.links.size(1),
                       (int)spec.links.size(2)},
                  stride_x{(int)spec.links.stride(0)},
                  background_reso{
                      spec.background_links.defined() ? (int)spec.background_links.size(1) : 0,
                  },
                  background_nlayers{
                      spec.background_data.defined() ? (int)spec.background_data.size(1) : 0},
                  basis_dim(spec.basis_dim),
                  sh_data_dim((int)spec.sh_data.size(1)),
                  basis_reso(spec.basis_data.defined() ? spec.basis_data.size(0) : 0),
                  _offset{spec._offset.data_ptr<float>()[0],
                          spec._offset.data_ptr<float>()[1],
                          spec._offset.data_ptr<float>()[2]},
                  _scaling{spec._scaling.data_ptr<float>()[0],
                           spec._scaling.data_ptr<float>()[1],
                           spec._scaling.data_ptr<float>()[2]}
            {
            }

            float *__restrict__ density_data;
            float *__restrict__ sh_data;
            const int32_t *__restrict__ links;

            const uint8_t basis_type;
            float *__restrict__ basis_data;

            const int32_t *__restrict__ background_links;
            float *__restrict__ background_data;

            const int size[3], stride_x;
            const int background_reso, background_nlayers;

            const int basis_dim, sh_data_dim, basis_reso;
            const float _offset[3];
            const float _scaling[3];
        };

        struct PackedTreeSpecLOT
        {
            PackedTreeSpecLOT(TreeSpecLOT &spec)
                : CornerD(spec.CornerD.data_ptr<float>()),
                  CornerSDF(spec.CornerSDF.data_ptr<float>()),
                  CornerGaussSDF(spec.CornerGaussSDF.data_ptr<float>()),
                  CornerSH(spec.CornerSH.data_ptr<float>()),
                  data(spec.data.data_ptr<float>()),
                  LearnS(spec.LearnS.data_ptr<float>()),
                  Beta(spec.Beta.data_ptr<float>()),

                  child(spec.child.data_ptr<int32_t>()),
                  NodeCorners(spec.NodeCorners.data_ptr<int32_t>()),
                  NodeAllNeighbors(spec.NodeAllNeighbors.data_ptr<int32_t>()),
                  NodeAllNeighLen(spec.NodeAllNeighLen.data_ptr<float>()),

                  basis_type(spec.basis_type),
                  basis_dim(spec.basis_dim),
                  sh_data_dim((int)spec.CornerSH.size(1)),
                  _offset{spec._offset.data_ptr<float>()[0],
                          spec._offset.data_ptr<float>()[1],
                          spec._offset.data_ptr<float>()[2]},
                  _scaling{spec._scaling.data_ptr<float>()[0],
                           spec._scaling.data_ptr<float>()[1],
                           spec._scaling.data_ptr<float>()[2]}
            {
            }

            float *__restrict__ CornerD;
            float *__restrict__ CornerSDF;
            float *__restrict__ CornerGaussSDF;
            float *__restrict__ CornerSH;
            float *__restrict__ data;
            float *__restrict__ LearnS;
            float *__restrict__ Beta;
            int32_t *__restrict__ child;
            int32_t *__restrict__ NodeCorners;
            int32_t *__restrict__ NodeAllNeighbors;
            float *__restrict__ NodeAllNeighLen;

            const uint8_t basis_type;

            const int basis_dim, sh_data_dim;
            const float _offset[3];
            const float _scaling[3];
        };

        struct PackedGridOutputGrads
        {
            PackedGridOutputGrads(GridOutputGrads &grads) : grad_density_out(grads.grad_density_out.defined() ? grads.grad_density_out.data_ptr<float>() : nullptr),
                                                            grad_sh_out(grads.grad_sh_out.defined() ? grads.grad_sh_out.data_ptr<float>() : nullptr),
                                                            grad_basis_out(grads.grad_basis_out.defined() ? grads.grad_basis_out.data_ptr<float>() : nullptr),
                                                            grad_background_out(grads.grad_background_out.defined() ? grads.grad_background_out.data_ptr<float>() : nullptr),
                                                            mask_out((grads.mask_out.defined() && grads.mask_out.size(0) > 0) ? grads.mask_out.data_ptr<bool>() : nullptr),
                                                            mask_background_out((grads.mask_background_out.defined() && grads.mask_background_out.size(0) > 0) ? grads.mask_background_out.data_ptr<bool>() : nullptr)
            {
            }
            float *__restrict__ grad_density_out;
            float *__restrict__ grad_sh_out;
            float *__restrict__ grad_basis_out;
            float *__restrict__ grad_background_out;

            bool *__restrict__ mask_out;
            bool *__restrict__ mask_background_out;
        };

        struct PackedGridOutputGradsLOT
        {
            PackedGridOutputGradsLOT(GridOutputGradsLOT &grads) : grad_density_out(grads.grad_density_out.defined() ? grads.grad_density_out.data_ptr<float>() : nullptr),
                                                                  grad_sh_out(grads.grad_sh_out.defined() ? grads.grad_sh_out.data_ptr<float>() : nullptr)
            {
            }
            float *__restrict__ grad_density_out;
            float *__restrict__ grad_sh_out;
        };

        struct PackedGridOutputGradsSDFLOT
        {
            PackedGridOutputGradsSDFLOT(GridOutputGradsSDFLOT &grads) : grad_sdf_out(grads.grad_sdf_out.defined() ? grads.grad_sdf_out.data_ptr<float>() : nullptr),
                                                                        grad_learns_out(grads.grad_sdf_out.defined() ? grads.grad_learns_out.data_ptr<float>() : nullptr),
                                                                        grad_beta_out(grads.grad_beta_out.defined() ? grads.grad_beta_out.data_ptr<float>() : nullptr),
                                                                        grad_sh_out(grads.grad_sh_out.defined() ? grads.grad_sh_out.data_ptr<float>() : nullptr)
            {
            }
            float *__restrict__ grad_sdf_out;
            float *__restrict__ grad_learns_out;
            float *__restrict__ grad_beta_out;
            float *__restrict__ grad_sh_out;
        };

        struct PackedCameraSpec
        {
            PackedCameraSpec(CameraSpec &cam) : c2w(cam.c2w.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
                                                fx(cam.fx), fy(cam.fy),
                                                cx(cam.cx), cy(cam.cy),
                                                width(cam.width), height(cam.height),
                                                ndc_coeffx(cam.ndc_coeffx), ndc_coeffy(cam.ndc_coeffy) {}
            const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
                c2w;
            float fx;
            float fy;
            float cx;
            float cy;
            int width;
            int height;

            float ndc_coeffx;
            float ndc_coeffy;
        };

        struct PackedRaysSpec
        {
            const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> origins;
            const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> dirs;
            PackedRaysSpec(RaysSpec &spec) : origins(spec.origins.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
                                             dirs(spec.dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>())
            {
            }
        };

        struct PackedRaysHitLOTreeSDF
        {
            PackedRaysHitLOTreeSDF(RaysHitLOTreeSDF &ray) : sdf_point(ray.sdf_point.data_ptr<float>()),
                                                            col_point(ray.col_point.data_ptr<float>()),
                                                            hitnode_sdf(ray.hitnode_sdf.data_ptr<int32_t>()),
                                                            hitnode_col(ray.hitnode_col.data_ptr<int32_t>()),
                                                            hitnum(ray.hitnum.data_ptr<int32_t>())
            {
            }

            float *__restrict__ sdf_point;
            float *__restrict__ col_point;
            int32_t *__restrict__ hitnode_sdf;
            int32_t *__restrict__ hitnode_col;
            int32_t *__restrict__ hitnum;
        };

        struct SingleRaySpec
        {
            SingleRaySpec() = default;
            __device__ SingleRaySpec(const float *__restrict__ origin, const float *__restrict__ dir)
                : origin{origin[0], origin[1], origin[2]},
                  dir{dir[0], dir[1], dir[2]} {}
            __device__ void set(const float *__restrict__ origin, const float *__restrict__ dir)
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    this->origin[i] = origin[i];
                    this->dir[i] = dir[i];
                }
            }

            float origin[3];
            float dir[3];
            float tmin, tmax, world_step;

            float pos[3];
            int32_t l[3];
            RandomEngine32 rng;

            float delta_scale;
            float invdir[3];
        };

        struct SingleRayHitLOTreeSDF
        {
            SingleRayHitLOTreeSDF() = default;
            __device__ SingleRayHitLOTreeSDF(float *__restrict__ sdf_point_in,
                                             float *__restrict__ col_point_in,
                                             int32_t *__restrict__ hitnode_sdf_in,
                                             int32_t *__restrict__ hitnode_col_in,
                                             const int32_t  hitnum_in)
                : sdf_point(sdf_point_in),
                  col_point(col_point_in),
                  hitnode_sdf(hitnode_sdf_in),
                  hitnode_col(hitnode_col_in),
                  hitnum(hitnum_in) {}
            __device__ void set(float *__restrict__ sdf_point_in,
                                float *__restrict__ col_point_in,
                                int32_t *__restrict__ hitnode_sdf_in,
                                int32_t *__restrict__ hitnode_col_in,
                                const int32_t hitnum_in)
            {
                this->sdf_point = sdf_point_in;
                this->col_point = col_point_in;
                this->hitnode_sdf = hitnode_sdf_in;
                this->hitnode_col = hitnode_col_in;
                this->hitnum = hitnum_in;
            }

            float *__restrict__ sdf_point;
            float *__restrict__ col_point;
            int32_t *__restrict__ hitnode_sdf;
            int32_t *__restrict__ hitnode_col;
            int32_t hitnum;
        };

    } // namespace device
} // namespace
