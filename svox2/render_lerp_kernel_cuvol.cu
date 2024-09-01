// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"

#include <iostream>
#include <cstdint>
#include <tuple>

namespace
{
    const int WARP_SIZE = 32;

    const int TRACE_RAY_CUDA_THREADS = 128;
    const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

    const int TRACE_RAY_BKWD_CUDA_THREADS = 128;
    const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

    const int MIN_BLOCKS_PER_SM = 8;

    const int TRACE_RAY_BG_CUDA_THREADS = 128;
    const int MIN_BG_BLOCKS_PER_SM = 8;
    typedef cub::WarpReduce<float> WarpReducef;

    namespace device
    {

        // * For ray rendering
        __device__ __inline__ void trace_ray_cuvol(
            const PackedSparseGridSpec &__restrict__ grid,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out,
            float *__restrict__ out_log_transmit)
        {
            const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
            const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

            if (ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = (grid.background_nlayers == 0) ? opt.background_brightness : 0.f;
                if (out_log_transmit != nullptr)
                {
                    *out_log_transmit = 0.f;
                }
                return;
            }

            float t = ray.tmin;
            float outv = 0.f;

            float log_transmit = 0.f;
            // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                    ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
                    ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
                    ray.pos[j] -= static_cast<float>(ray.l[j]);
                }

                const float skip = compute_skip_dist(ray,
                                                     grid.links, grid.stride_x,
                                                     grid.size[2], 0);

                if (skip >= opt.step_size)
                {
                    // For consistency, we skip the by step size
                    t += ceilf(skip / opt.step_size) * opt.step_size;
                    continue;
                }
                float sigma = trilerp_cuvol_one(
                    grid.links, grid.density_data,
                    grid.stride_x,
                    grid.size[2],
                    1,
                    ray.l, ray.pos,
                    0);
                if (opt.last_sample_opaque && t + opt.step_size > ray.tmax)
                {
                    ray.world_step = 1e9;
                }
                // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;

                if (sigma > opt.sigma_thresh)
                {
                    float lane_color = trilerp_cuvol_one(
                        grid.links,
                        grid.sh_data,
                        grid.stride_x,
                        grid.size[2],
                        grid.sh_data_dim,
                        ray.l, ray.pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = ray.world_step * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += opt.step_size;
            }

            if (grid.background_nlayers == 0)
            {
                outv += _EXP(log_transmit) * opt.background_brightness;
            }
            if (lane_colorgrp_id == 0)
            {
                if (out_log_transmit != nullptr)
                {
                    *out_log_transmit = log_transmit;
                }
                out[lane_colorgrp] = outv;
            }
        }

        // * For ray rendering
        __device__ __inline__ void trace_ray_cuvol_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out,
            float *__restrict__ out_log_transmit)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            const int tree_N = 2;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;
                if (out_log_transmit != nullptr)
                {
                    *out_log_transmit = 0.f;
                }
                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sigma = trilerp_cuvol_one_LOT(node_corner_ids,
                                                    lotree.CornerD,
                                                    1, pos, 0);

                float att;
                float subcube_tmin, subcube_tmax;
                _dda_unit_LOT(pos, ray.invdir, &subcube_tmin, &subcube_tmax);

                const float t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const float delta_t = t_subcube + opt.step_size;

                if (sigma > opt.sigma_thresh)
                {
                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                if (out_log_transmit != nullptr)
                {
                    *out_log_transmit = log_transmit;
                }
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_convert_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

         __device__ __inline__ void trace_ray_volsdf_downsample_test_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            int32_t *__restrict__ leaf_node_map,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int64_t node_id;
                query_single_from_root_downsample_LOT(lotree.child,
                                                      lotree.NodeCorners, pos,
                                                      &cube_sz, &node_id);
                int32_t node_index = leaf_node_map[node_id];

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = lotree.CornerSDF[node_index];
                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    int64_t sh_index = int64_t(node_index * lotree.sh_data_dim) + lane_id;
                    float lane_color = lotree.CornerSH[sh_index];
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    //outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_downsample_render_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int64_t node_id;
                query_single_from_root_downsample_LOT(lotree.child,
                                                      lotree.NodeCorners, pos,
                                                      &cube_sz, &node_id);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    int64_t sdf_index = int64_t(node_id * (lotree.sh_data_dim + 1)) + lotree.sh_data_dim;
                    float sdf = lotree.data[sdf_index];
                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    int64_t sh_index = int64_t(node_id * (lotree.sh_data_dim + 1)) + lane_id;
                    float lane_color = lotree.data[sh_index];
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    //outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_downsample_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            const int32_t *__restrict__ leaf_node_map,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;
            const uint32_t leader_mask = 1U | (1U << lotree.basis_dim) | (1U << (2 * lotree.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            const int tree_N = 2;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float t = ray.tmin;

            const float gout = grad_output[lane_colorgrp];

            float log_transmit = 0.f;
            float cube_sz;

            // float last_beta_grad = 0.0;

            // remat samples
            float pos[3];
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int64_t node_id;
                query_single_from_root_downsample_LOT(lotree.child,
                                                      lotree.NodeCorners, pos,
                                                      &cube_sz, &node_id);
                int32_t node_index = leaf_node_map[node_id];

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = lotree.CornerSDF[node_index];
                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    int64_t sh_index = int64_t(node_index * lotree.sh_data_dim) + lane_id;
                    float lane_color = lotree.CornerSH[sh_index];
                    float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));

                    const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                    float total_color = fmaxf(lane_color_total, 0.f);
                    float color_in_01 = total_color == lane_color_total;
                    total_color *= gout; // Clamp to [+0, infty)

                    float total_color_c1 = __shfl_sync(leader_mask, total_color, lotree.basis_dim);
                    total_color += __shfl_sync(leader_mask, total_color, 2 * lotree.basis_dim);
                    total_color += total_color_c1;

                    color_in_01 = __shfl_sync((1U << lotree.sh_data_dim) - 1, color_in_01, lane_colorgrp * lotree.basis_dim);
                    const float grad_common = weight * color_in_01 * gout;
                    const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                    log_transmit -= pcnt;

                    accum -= weight * total_color;
                    float curr_grad_sdf = delta_t * ray.delta_scale * (total_color * _EXP(log_transmit) - accum) *
                                          (-1.0 / (2.0 * lotree.Beta[0] * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0]));

                    atomicAdd(&grads.grad_sh_out[sh_index], curr_grad_color);

                    if (lane_id == 0)
                    {
                        atomicAdd(&grads.grad_sdf_out[node_index], curr_grad_sdf);
                    }

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        break;
                    }

                }
                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_presample_volsdf_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ presample_t,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = presample_t[0];

            while (t < presample_t[1])
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                  lotree.CornerSDF,
                                                  1, pos, 0);

                float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                float lane_color = trilerp_cuvol_one_LOT(
                    node_corner_ids,
                    lotree.CornerSH,
                    lotree.sh_data_dim, pos, lane_id);
                lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                const float delta_t = presample_t[2];

                const float pcnt = delta_t * ray.delta_scale * sigma;
                const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                log_transmit -= pcnt;

                float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)

                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_gaussian_volsdf_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerGaussSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_gaussian_presample_volsdf_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ presample_t,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = presample_t[0];
            while (t < presample_t[1])
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                  lotree.CornerGaussSDF,
                                                  1, pos, 0);

                float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                float lane_color = trilerp_cuvol_one_LOT(
                    node_corner_ids,
                    lotree.CornerSH,
                    lotree.sh_data_dim, pos, lane_id);
                lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                const float delta_t = presample_t[2];

                const float pcnt = delta_t * ray.delta_scale * sigma;
                const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                log_transmit -= pcnt;

                float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)

                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_record_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            int count = 0;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                float pos_unchange[3];
                for (int j = 0; j < 3; ++j)
                {
                    pos_unchange[j] = pos[j];
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    if (lane_id == 0)
                    {
                        out[count * 3] = pos_unchange[0];
                        out[count * 3 + 1] = pos_unchange[1];
                        out[count * 3 + 2] = pos_unchange[2];
                    }

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }

                    count++;
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_record_w_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            int count = 0;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                float pos_unchange[3];
                for (int j = 0; j < 3; ++j)
                {
                    pos_unchange[j] = pos[j];
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];
                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    if (lane_id == 0)
                    {
                        out[count * 9] = sdf;
                        out[count * 9 + 1] = weight;
                        out[count * 9 + 2] = log_transmit;
                        out[count * 9 + 3] = sigma;
                        out[count * 9 + 4] = delta_t;
                        out[count * 9 + 5] = cube_sz;
                        out[count * 9 + 6] = ray.delta_scale;
                        out[count * 9 + 7] = pcnt;
                        out[count * 9 + 8] = pcnt;
                    }

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }

                    count++;
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_presample_volsdf_record_w_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ presample_t,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            int count = 0;
            float t = presample_t[0];
            while (t < presample_t[1])
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                  lotree.CornerSDF,
                                                  1, pos, 0);

                float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                const float delta_t = presample_t[2];

                const float pcnt = delta_t * ray.delta_scale * sigma;
                const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                log_transmit -= pcnt;

                if (lane_id == 0)
                {
                    out[count * 9] = sdf;
                    out[count * 9 + 1] = weight;
                    out[count * 9 + 2] = log_transmit;
                    out[count * 9 + 3] = sigma;
                    out[count * 9 + 4] = delta_t;
                    out[count * 9 + 5] = cube_sz;
                    out[count * 9 + 6] = ray.delta_scale;
                    out[count * 9 + 7] = pcnt;
                    out[count * 9 + 8] = presample_t[0];
                }

                count++;

                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_eikonal_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            const float scale,
            float *__restrict__ uni_sample,
            float *__restrict__ total_eikonal_loss,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float cube_sz;
            float log_transmit = 0.f;

            float pos[3];
            float t = ray.tmin;

            bool ray_sample = true;
            bool uni_not_sample = true;

            while (uni_not_sample)
            {
                if (ray_sample)
                {
#pragma unroll 3
                    for (int j = 0; j < 3; ++j)
                    {
                        pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                    }
                }
                else
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        pos[j] = uni_sample[j];
                    }

                    uni_not_sample = false;
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                  lotree.CornerSDF,
                                                  1, pos, 0);

                float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                float delta_t;
                if (ray_sample)
                {
                    float att;
                    float subcube_tmin, subcube_tmax;
                    _dda_unit_LOT(pos, ray.invdir, &subcube_tmin, &subcube_tmax);

                    const float t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                    delta_t = t_subcube + opt.step_size;
                    t += delta_t;
                }

                if (((sigma > opt.sigma_thresh) && ray_sample) || !uni_not_sample)
                {
                    float Dx_t = trilerp_eikonal_one_LOT(node_corner_ids,
                                                         lotree.CornerSDF,
                                                         0, 1, pos, 0);

                    float Dy_t = trilerp_eikonal_one_LOT(node_corner_ids,
                                                         lotree.CornerSDF,
                                                         1, 1, pos, 0);

                    float Dz_t = trilerp_eikonal_one_LOT(node_corner_ids,
                                                         lotree.CornerSDF,
                                                         2, 1, pos, 0);

                    float sdf_grad_len = sqrt(Dx_t * Dx_t + Dy_t * Dy_t + Dz_t * Dz_t + 1e-8);

                    float eikonal_loss_sqrt = sdf_grad_len - 1.0;
                    float eikonal_loss = eikonal_loss_sqrt * eikonal_loss_sqrt;

                    float sdf_grad_len_inv = 1.0 / sdf_grad_len;

                    float grad_x = eikonal_loss_sqrt * sdf_grad_len_inv * 2.0 * scale * Dx_t;
                    trilerp_backward_eikonal_LOT(node_corner_ids, 0, grads.grad_sdf_out, pos, grad_x);

                    float grad_y = eikonal_loss_sqrt * sdf_grad_len_inv * 2.0 * scale * Dy_t;
                    trilerp_backward_eikonal_LOT(node_corner_ids, 1, grads.grad_sdf_out, pos, grad_y);

                    float grad_z = eikonal_loss_sqrt * sdf_grad_len_inv * 2.0 * scale * Dz_t;
                    trilerp_backward_eikonal_LOT(node_corner_ids, 2, grads.grad_sdf_out, pos, grad_z);

                    if (ray_sample)
                    {
                        const float pcnt = delta_t * ray.delta_scale * sigma;
                        const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                        log_transmit -= pcnt;

                        if (_EXP(log_transmit) < opt.stop_thresh)
                        {
                            ray_sample = false;
                        }
                    }

                    total_eikonal_loss[0] += eikonal_loss;
                }

                if (t > ray.tmax)
                {
                    ray_sample = false;
                }
            }
        }

        __device__ __inline__ void trace_ray_sdf_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRayHitLOTreeSDF &__restrict__ ray_hls,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray_hls.hitnum <= 1)
            {
                out[lane_colorgrp] = opt.background_brightness;
                return;
            }

            float outv = 0.f;
            float accum_trans = 1.0;

            float sdf_p = 0.0;

            float c_sdf_point_p[3];
            for (int j = 0; j < 3; ++j)
            {
                c_sdf_point_p[j] = ray_hls.sdf_point[j];
            }

            int32_t node_corner_ids_p[8];
            for (int j = 0; j < 8; ++j)
            {
                node_corner_ids_p[j] = ray_hls.hitnode_sdf[j];
            }

            sdf_p = trilerp_cuvol_one_LOT(node_corner_ids_p,
                                          lotree.CornerSDF,
                                          1, c_sdf_point_p, 0);

            for (int on = 0; on < ray_hls.hitnum - 1; ++on)
            {
                float c_sdf_point_n[3];
                for (int j = 0; j < 3; ++j)
                {
                    c_sdf_point_n[j] = ray_hls.sdf_point[(on + 1) * 3 + j];
                }

                int32_t node_corner_ids_n[8];
                for (int j = 0; j < 8; ++j)
                {
                    node_corner_ids_n[j] = ray_hls.hitnode_sdf[(on + 1) * 8 + j];
                }

                float sdf_n = trilerp_cuvol_one_LOT(node_corner_ids_n,
                                                    lotree.CornerSDF,
                                                    1, c_sdf_point_n, 0);

                float cdf_p = 1.0 / (1.0 + expf(-lotree.LearnS[0] * sdf_p));
                float cdf_n = 1.0 / (1.0 + expf(-lotree.LearnS[0] * sdf_n));

                float alpha = min(max((cdf_p - cdf_n + 1e-5) / (cdf_p + 1e-5), 0.0), 1.0);

                if (alpha > opt.alpha_thresh)
                {
                    float c_col_point[3];
                    for (int j = 0; j < 3; ++j)
                    {
                        c_col_point[j] = ray_hls.col_point[on * 3 + j];
                    }

                    int32_t node_corner_ids[8];
                    for (int j = 0; j < 8; ++j)
                    {
                        node_corner_ids[j] = ray_hls.hitnode_col[on * 8 + j];
                    }

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, c_col_point, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id];

                    const float weight = accum_trans * alpha;
                    accum_trans *= (1.0 - alpha + 1e-7);

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)

                    if (accum_trans < opt.stop_thresh)
                    {
                        accum_trans = -1e3f;
                        break;
                    }
                }

                sdf_p = sdf_n;
            }

            outv += accum_trans * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_sdf_grad_LOT(
            float *__restrict__ corner_sdf_grad,
            float *__restrict__ sdf_point,
            int *__restrict__ hitnode_sdf,
            int32_t hitnum,
            float *__restrict__ sdf_grad_out)
        {
            for (int on = 0; on < hitnum; ++on)
            {
                float c_sdf_point[3];
                for (int j = 0; j < 3; ++j)
                {
                    c_sdf_point[j] = sdf_point[on * 3 + j];
                }

                int32_t node_corner_ids[8];
                for (int j = 0; j < 8; ++j)
                {
                    node_corner_ids[j] = hitnode_sdf[on * 8 + j];
                }

                float sdf_grad_x = trilerp_cuvol_one_LOT(node_corner_ids,
                                                         corner_sdf_grad,
                                                         3, c_sdf_point, 0);

                float sdf_grad_y = trilerp_cuvol_one_LOT(node_corner_ids,
                                                         corner_sdf_grad,
                                                         3, c_sdf_point, 1);

                float sdf_grad_z = trilerp_cuvol_one_LOT(node_corner_ids,
                                                         corner_sdf_grad,
                                                         3, c_sdf_point, 2);

                sdf_grad_out[on * 3] = sdf_grad_x;
                sdf_grad_out[on * 3 + 1] = sdf_grad_y;
                sdf_grad_out[on * 3 + 2] = sdf_grad_z;
            }
        }

        __device__ __inline__ void trace_ray_hitnum_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            int *__restrict__ hit_num)
        {
            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sigma = trilerp_cuvol_one_LOT(node_corner_ids,
                                                    lotree.CornerD,
                                                    1, pos, 0);

                float att;
                float subcube_tmin, subcube_tmax;
                _dda_unit_LOT(pos, ray.invdir, &subcube_tmin, &subcube_tmax);

                const float t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const float delta_t = t_subcube + opt.step_size;

                if (sigma > opt.sigma_thresh)
                {
                    hit_num[0] += 1;

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    log_transmit -= pcnt;

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_hitnum_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            int *__restrict__ hit_num)
        {
            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    hit_num[0] += 1;

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    log_transmit -= pcnt;

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_colrefine_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            float ray_mse,
            const RenderOptions &__restrict__ opt,
            float *__restrict__ node_mse)
        {
            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;
            const int tree_N = 2;

            float pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                int32_t node_id[4];
                query_single_from_root_refine_LOT(lotree.child,
                                                  lotree.NodeCorners, pos,
                                                  &cube_sz, node_corner_ids, node_id);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    log_transmit -= pcnt;

                    int64_t node_flat_id = node_id[0] * tree_N * tree_N * tree_N + node_id[1] * tree_N * tree_N + node_id[2] * tree_N + node_id[3];

                    atomicMax(&node_mse[node_flat_id], ray_mse);

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_prehit_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            float *__restrict__ hit_info)
        {
            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float cube_sz;

            float pos[3];
            float t = ray.tmin;

            bool hit_start = false;
            bool hit_mid = false;
            bool hit_end = false;
            float last_sdf = 999.0;
            float last_t = ray.tmin;

            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                  lotree.CornerSDF,
                                                  1, pos, 0);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (sdf > 0.0)
                {
                    if (sdf < opt.sdf_thresh && !hit_start)
                    {
                        hit_start = true;
                        hit_info[0] = t;
                    }

                    if (sdf > last_sdf && hit_start)
                    {
                        hit_start = false;
                        hit_info[0] = ray.tmin;
                    }

                    if (hit_mid && !hit_end)
                    {
                        hit_end = true;
                        hit_info[1] = last_t;
                    }
                }
                else if (sdf < 0.0)
                {
                    if (!hit_mid)
                    {
                        hit_mid = true;
                    }

                    if (!hit_start)
                    {
                        hit_start = true;
                        hit_info[0] = last_t;
                    }

                    if (-sdf > opt.sdf_thresh)
                    {
                        hit_end = true;
                        hit_info[1] = t;
                    }
                    else
                    {
                        if (sdf > last_sdf)
                        {
                            hit_end = true;
                            hit_info[1] = t;
                        }
                    }
                }

                if (hit_start && hit_end)
                {
                    break;
                }

                last_sdf = sdf;
                last_t = t;

                t += delta_t;
            }

            if (hit_info[0] == 0.0)
            {
                hit_info[0] = ray.tmin;
            }
            if (hit_info[1] == 0.0)
            {
                hit_info[1] = ray.tmax;
            }

            hit_info[2] = (hit_info[1] - hit_info[0]) / (opt.sample_size - 1);
        }

        __device__ __inline__ void trace_ray_hitpoint_sdf_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            float step,
            float *__restrict__ sdf_point_out,
            float *__restrict__ col_point_out,
            int32_t *__restrict__ hitnode_sdf_out,
            int32_t *__restrict__ hitnode_col_out,
            int32_t *__restrict__ hitnum_out)
        {
            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float cube_sz;

            int order_num = 0;

            float pos[3];
            float prev_pos[3];
            float t = ray.tmin;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                float col_pos[3];
                if (order_num > 0)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        col_pos[j] = (prev_pos[j] + pos[j]) / 2.0;
                    }
                }

                for (int j = 0; j < 3; ++j)
                {
                    prev_pos[j] = pos[j];
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                for (int j = 0; j < 3; ++j)
                {
                    sdf_point_out[order_num * 3 + j] = pos[j];
                }
                for (int j = 0; j < 8; ++j)
                {
                    hitnode_sdf_out[order_num * 8 + j] = node_corner_ids[j];
                }

                if (order_num > 0)
                {
                    query_single_from_root_LOT(lotree.child,
                                               lotree.NodeCorners, col_pos,
                                               &cube_sz, node_corner_ids);

                    for (int j = 0; j < 3; ++j)
                    {
                        col_point_out[(order_num - 1) * 3 + j] = col_pos[j];
                    }
                    for (int j = 0; j < 8; ++j)
                    {
                        hitnode_col_out[(order_num - 1) * 8 + j] = node_corner_ids[j];
                    }
                }

                float att;
                float subcube_tmin, subcube_tmax;
                _dda_unit_LOT(pos, ray.invdir, &subcube_tmin, &subcube_tmax);

                const float t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const float delta_t = t_subcube + step;

                t += delta_t;

                order_num += 1;
            }

            hitnum_out[0] = order_num;
        }

        __device__ __inline__ void trace_ray_refine_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ color_out,
            int *__restrict__ hitnode_out,
            float *__restrict__ weight_out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            const int tree_N = 2;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                color_out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;

            float pos[3];
            float t = ray.tmin;
            int hit_num = 0;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                int32_t node_id[4];
                query_single_from_root_refine_LOT(lotree.child,
                                                  lotree.NodeCorners, pos,
                                                  &cube_sz, node_corner_ids, node_id);

                float sigma = trilerp_cuvol_one_LOT(node_corner_ids,
                                                    lotree.CornerD,
                                                    1, pos, 0);

                float att;
                float subcube_tmin, subcube_tmax;
                _dda_unit_LOT(pos, ray.invdir, &subcube_tmin, &subcube_tmax);

                const float t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const float delta_t = t_subcube + opt.step_size;

                if (sigma > opt.sigma_thresh)
                {
                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)

                    if (lane_id == 0)
                    {
                        int64_t node_flat_id = node_id[0] * tree_N * tree_N * tree_N + node_id[1] * tree_N * tree_N + node_id[2] * tree_N + node_id[3];
                        atomicMax(&weight_out[node_flat_id], weight);

                        for (int node_i = 0; node_i < 4; ++node_i)
                        {
                            hitnode_out[hit_num * 4 + node_i] = node_id[node_i];
                        }
                    }

                    hit_num += 1;

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                color_out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_refine_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ color_out,
            int *__restrict__ hitnode_out,
            float *__restrict__ weight_out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                color_out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;
            const int tree_N = 2;

            float pos[3];
            float t = ray.tmin;
            int hit_num = 0;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                int32_t node_id[4];
                query_single_from_root_refine_LOT(lotree.child,
                                                  lotree.NodeCorners, pos,
                                                  &cube_sz, node_corner_ids, node_id);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)

                    if (lane_id == 0)
                    {
                        int64_t node_flat_id = node_id[0] * tree_N * tree_N * tree_N + node_id[1] * tree_N * tree_N + node_id[2] * tree_N + node_id[3];
                        atomicMax(&weight_out[node_flat_id], weight);

                        for (int node_i = 0; node_i < 4; ++node_i)
                        {
                            hitnode_out[hit_num * 4 + node_i] = node_id[node_i];
                        }
                    }

                    hit_num += 1;

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                color_out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_refine_sdf_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            float *__restrict__ sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float *__restrict__ color_out,
            int *__restrict__ hitnode_out,
            float *__restrict__ sdf_out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                color_out[lane_colorgrp] = opt.background_brightness;

                return;
            }

            float outv = 0.f;
            float log_transmit = 0.f;
            float cube_sz;
            const int tree_N = 2;

            float pos[3];
            float t = ray.tmin;
            int hit_num = 0;
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                int32_t node_id[4];
                query_single_from_root_refine_LOT(lotree.child,
                                                  lotree.NodeCorners, pos,
                                                  &cube_sz, node_corner_ids, node_id);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(lane_color, lane_colorgrp_id == 0);
                    outv += weight * fmaxf(lane_color_total + 0.5f, 0.f); // Clamp to [+0, infty)

                    if (lane_id == 0)
                    {
                        int64_t node_flat_id = node_id[0] * tree_N * tree_N * tree_N + node_id[1] * tree_N * tree_N + node_id[2] * tree_N + node_id[3];

                        float sdf_abs = fabsf(sdf);

                        atomicMin(&sdf_out[node_flat_id], sdf_abs);

                        for (int node_i = 0; node_i < 4; ++node_i)
                        {
                            hitnode_out[hit_num * 4 + node_i] = node_id[node_i];
                        }
                    }

                    hit_num += 1;

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += delta_t;
            }

            outv += _EXP(log_transmit) * opt.background_brightness;
            if (lane_colorgrp_id == 0)
            {
                color_out[lane_colorgrp] = outv;
            }
        }

        __device__ __inline__ void trace_ray_expected_term(
            const PackedSparseGridSpec &__restrict__ grid,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            float *__restrict__ out)
        {
            if (ray.tmin > ray.tmax)
            {
                *out = 0.f;
                return;
            }

            float t = ray.tmin;
            float outv = 0.f;

            float log_transmit = 0.f;
            // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                    ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
                    ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
                    ray.pos[j] -= static_cast<float>(ray.l[j]);
                }

                const float skip = compute_skip_dist(ray,
                                                     grid.links, grid.stride_x,
                                                     grid.size[2], 0);

                if (skip >= opt.step_size)
                {
                    // For consistency, we skip the by step size
                    t += ceilf(skip / opt.step_size) * opt.step_size;
                    continue;
                }
                float sigma = trilerp_cuvol_one(
                    grid.links, grid.density_data,
                    grid.stride_x,
                    grid.size[2],
                    1,
                    ray.l, ray.pos,
                    0);
                if (sigma > opt.sigma_thresh)
                {
                    const float pcnt = ray.world_step * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    outv += weight * (t / opt.step_size) * ray.world_step;
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        log_transmit = -1e3f;
                        break;
                    }
                }
                t += opt.step_size;
            }
            *out = outv;
        }

        // From Dex-NeRF
        __device__ __inline__ void trace_ray_sigma_thresh(
            const PackedSparseGridSpec &__restrict__ grid,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            float sigma_thresh,
            float *__restrict__ out)
        {
            if (ray.tmin > ray.tmax)
            {
                *out = 0.f;
                return;
            }

            float t = ray.tmin;
            *out = 0.f;

            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                    ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
                    ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
                    ray.pos[j] -= static_cast<float>(ray.l[j]);
                }

                const float skip = compute_skip_dist(ray,
                                                     grid.links, grid.stride_x,
                                                     grid.size[2], 0);

                if (skip >= opt.step_size)
                {
                    // For consistency, we skip the by step size
                    t += ceilf(skip / opt.step_size) * opt.step_size;
                    continue;
                }
                float sigma = trilerp_cuvol_one(
                    grid.links, grid.density_data,
                    grid.stride_x,
                    grid.size[2],
                    1,
                    ray.l, ray.pos,
                    0);
                if (sigma > sigma_thresh)
                {
                    *out = (t / opt.step_size) * ray.world_step;
                    break;
                }
                t += opt.step_size;
            }
        }

        __device__ __inline__ void trace_ray_cuvol_backward(
            const PackedSparseGridSpec &__restrict__ grid,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            const float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float log_transmit_in,
            float beta_loss,
            float sparsity_loss,
            PackedGridOutputGrads &__restrict__ grads,
            float *__restrict__ accum_out,
            float *__restrict__ log_transmit_out)
        {
            const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
            const uint32_t lane_colorgrp = lane_id / grid.basis_dim;
            const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            if (ray.tmin > ray.tmax)
            {
                if (accum_out != nullptr)
                {
                    *accum_out = accum;
                }
                if (log_transmit_out != nullptr)
                {
                    *log_transmit_out = 0.f;
                }
                // printf("accum_end_fg_fast=%f\n", accum);
                return;
            }

            if (beta_loss > 0.f)
            {
                const float transmit_in = _EXP(log_transmit_in);
                beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
                accum += beta_loss;
                // Interesting how this loss turns out, kinda nice?
            }

            float t = ray.tmin;

            const float gout = grad_output[lane_colorgrp];

            float log_transmit = 0.f;

            // remat samples
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                    ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
                    ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
                    ray.pos[j] -= static_cast<float>(ray.l[j]);
                }
                const float skip = compute_skip_dist(ray,
                                                     grid.links, grid.stride_x,
                                                     grid.size[2], 0);
                if (skip >= opt.step_size)
                {
                    // For consistency, we skip the by step size
                    t += ceilf(skip / opt.step_size) * opt.step_size;
                    continue;
                }

                float sigma = trilerp_cuvol_one(
                    grid.links,
                    grid.density_data,
                    grid.stride_x,
                    grid.size[2],
                    1,
                    ray.l, ray.pos,
                    0);
                if (opt.last_sample_opaque && t + opt.step_size > ray.tmax)
                {
                    ray.world_step = 1e9;
                }
                // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;
                if (sigma > opt.sigma_thresh)
                {
                    float lane_color = trilerp_cuvol_one(
                        grid.links,
                        grid.sh_data,
                        grid.stride_x,
                        grid.size[2],
                        grid.sh_data_dim,
                        ray.l, ray.pos, lane_id);
                    float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                    const float pcnt = ray.world_step * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                    float total_color = fmaxf(lane_color_total, 0.f);
                    float color_in_01 = total_color == lane_color_total;
                    total_color *= gout; // Clamp to [+0, infty)

                    float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim);
                    total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim);
                    total_color += total_color_c1;

                    color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.basis_dim);
                    const float grad_common = weight * color_in_01 * gout;
                    const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                    if (grid.basis_type != BASIS_TYPE_SH)
                    {
                        float curr_grad_sphfunc = lane_color * grad_common;
                        const float curr_grad_up2 = __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                                                                     curr_grad_sphfunc, 2 * grid.basis_dim);
                        curr_grad_sphfunc += __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                                                              curr_grad_sphfunc, grid.basis_dim);
                        curr_grad_sphfunc += curr_grad_up2;
                        if (lane_id < grid.basis_dim)
                        {
                            grad_sphfunc_val[lane_id] += curr_grad_sphfunc;
                        }
                    }

                    accum -= weight * total_color;
                    float curr_grad_sigma = ray.world_step * (total_color * _EXP(log_transmit) - accum);
                    if (sparsity_loss > 0.f)
                    {
                        // Cauchy version (from SNeRG)
                        curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                        // Alphs version (from PlenOctrees)
                        // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
                    }
                    trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out,
                                               grid.stride_x,
                                               grid.size[2],
                                               grid.sh_data_dim,
                                               ray.l, ray.pos,
                                               curr_grad_color, lane_id);
                    if (lane_id == 0)
                    {
                        trilerp_backward_cuvol_one_density(
                            grid.links,
                            grads.grad_density_out,
                            grads.mask_out,
                            grid.stride_x,
                            grid.size[2],
                            ray.l, ray.pos, curr_grad_sigma);
                    }
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        break;
                    }
                }
                t += opt.step_size;
            }
            if (lane_id == 0)
            {
                if (accum_out != nullptr)
                {
                    // Cancel beta loss out in case of background
                    accum -= beta_loss;
                    *accum_out = accum;
                }
                if (log_transmit_out != nullptr)
                {
                    *log_transmit_out = log_transmit;
                }
                // printf("accum_end_fg=%f\n", accum);
                // printf("log_transmit_fg=%f\n", log_transmit);
            }
        }

        __device__ __inline__ void trace_ray_cuvol_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            const float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            float log_transmit_in,
            float beta_loss,
            float sparsity_loss,
            PackedGridOutputGradsLOT &__restrict__ grads,
            float *__restrict__ accum_out,
            float *__restrict__ log_transmit_out)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;
            const uint32_t leader_mask = 1U | (1U << lotree.basis_dim) | (1U << (2 * lotree.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            const int tree_N = 2;

            if (ray.tmin > ray.tmax)
            {
                if (accum_out != nullptr)
                {
                    *accum_out = accum;
                }
                if (log_transmit_out != nullptr)
                {
                    *log_transmit_out = 0.f;
                }
                // printf("accum_end_fg_fast=%f\n", accum);
                return;
            }

            if (beta_loss > 0.f)
            {
                const float transmit_in = _EXP(log_transmit_in);
                beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
                accum += beta_loss;
                // Interesting how this loss turns out, kinda nice?
            }

            float t = ray.tmin;

            const float gout = grad_output[lane_colorgrp];

            float log_transmit = 0.f;
            float cube_sz;

            // remat samples
            float pos[3];
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sigma = trilerp_cuvol_one_LOT(node_corner_ids,
                                                    lotree.CornerD,
                                                    1, pos, 0);

                float att;
                float subcube_tmin, subcube_tmax;
                _dda_unit_LOT(pos, ray.invdir, &subcube_tmin, &subcube_tmax);

                const float t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const float delta_t = t_subcube + opt.step_size;

                // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;
                if (sigma > opt.sigma_thresh)
                {
                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                    float total_color = fmaxf(lane_color_total, 0.f);
                    float color_in_01 = total_color == lane_color_total;
                    total_color *= gout; // Clamp to [+0, infty)

                    float total_color_c1 = __shfl_sync(leader_mask, total_color, lotree.basis_dim);
                    total_color += __shfl_sync(leader_mask, total_color, 2 * lotree.basis_dim);
                    total_color += total_color_c1;

                    color_in_01 = __shfl_sync((1U << lotree.sh_data_dim) - 1, color_in_01, lane_colorgrp * lotree.basis_dim);
                    const float grad_common = weight * color_in_01 * gout;
                    const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                    accum -= weight * total_color;
                    float curr_grad_sigma = delta_t * ray.delta_scale * (total_color * _EXP(log_transmit) - accum);

                    if (sparsity_loss > 0.f)
                    {
                        // Cauchy version (from SNeRG)
                        curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                        // Alphs version (from PlenOctrees)
                        // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
                    }

                    trilerp_backward_cuvol_one_LOT(node_corner_ids, grads.grad_sh_out,
                                                   lotree.sh_data_dim,
                                                   pos,
                                                   curr_grad_color, lane_id);
                    if (lane_id == 0)
                    {
                        trilerp_backward_cuvol_one_density_LOT(
                            node_corner_ids,
                            grads.grad_density_out,
                            pos, curr_grad_sigma);
                    }
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        break;
                    }
                }
                t += delta_t;
            }
            if (lane_id == 0)
            {
                if (accum_out != nullptr)
                {
                    // Cancel beta loss out in case of background
                    accum -= beta_loss;
                    *accum_out = accum;
                }
                if (log_transmit_out != nullptr)
                {
                    *log_transmit_out = log_transmit;
                }
                // printf("accum_end_fg=%f\n", accum);
                // printf("log_transmit_fg=%f\n", log_transmit);
            }
        }

        __device__ __inline__ void trace_ray_volsdf_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            const float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;
            const uint32_t leader_mask = 1U | (1U << lotree.basis_dim) | (1U << (2 * lotree.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            const int tree_N = 2;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float t = ray.tmin;

            const float gout = grad_output[lane_colorgrp];

            float log_transmit = 0.f;
            float cube_sz;

            // float last_beta_grad = 0.0;

            // remat samples
            float pos[3];
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));

                    const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                    float total_color = fmaxf(lane_color_total, 0.f);
                    float color_in_01 = total_color == lane_color_total;
                    total_color *= gout; // Clamp to [+0, infty)

                    float total_color_c1 = __shfl_sync(leader_mask, total_color, lotree.basis_dim);
                    total_color += __shfl_sync(leader_mask, total_color, 2 * lotree.basis_dim);
                    total_color += total_color_c1;

                    color_in_01 = __shfl_sync((1U << lotree.sh_data_dim) - 1, color_in_01, lane_colorgrp * lotree.basis_dim);
                    const float grad_common = weight * color_in_01 * gout;
                    const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                    // float beta_grad_0 = -delta_t * ray.delta_scale * _EXP(-pcnt) *
                    //                     (((sdf / (2.0 * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0])) - sigma * lotree.Beta[0]) / (lotree.Beta[0] * lotree.Beta[0]));
                    // float beta_grad_1 = last_beta_grad * _EXP(-pcnt) + beta_grad_0 * _EXP(log_transmit);
                    // float curr_beta_grad = (last_beta_grad - beta_grad_1) * total_color;

                    // last_beta_grad = beta_grad_1;

                    log_transmit -= pcnt;

                    accum -= weight * total_color;
                    float curr_grad_sdf = delta_t * ray.delta_scale * (total_color * _EXP(log_transmit) - accum) *
                                          (-1.0 / (2.0 * lotree.Beta[0] * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0]));

                    trilerp_backward_cuvol_one_LOT(node_corner_ids, grads.grad_sh_out,
                                                   lotree.sh_data_dim,
                                                   pos,
                                                   curr_grad_color, lane_id);
                    if (lane_id == 0)
                    {
                        trilerp_backward_cuvol_one_density_LOT(
                            node_corner_ids,
                            grads.grad_sdf_out,
                            pos, curr_grad_sdf);

                        // atomicAdd(&grads.grad_beta_out[0], curr_beta_grad);
                    }
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        break;
                    }
                }
                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_volsdf_convert_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            const float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;
            const uint32_t leader_mask = 1U | (1U << lotree.basis_dim) | (1U << (2 * lotree.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            const int tree_N = 2;

            if (ray.tmax < 0 || ray.tmin > ray.tmax)
            {
                return;
            }

            float t = ray.tmin;

            const float gout = grad_output[lane_colorgrp];

            float log_transmit = 0.f;
            float cube_sz;

            // float last_beta_grad = 0.0;

            // remat samples
            float pos[3];
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));

                    const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                    float total_color = fmaxf(lane_color_total, 0.f);
                    float color_in_01 = total_color == lane_color_total;
                    total_color *= gout; // Clamp to [+0, infty)

                    float total_color_c1 = __shfl_sync(leader_mask, total_color, lotree.basis_dim);
                    total_color += __shfl_sync(leader_mask, total_color, 2 * lotree.basis_dim);
                    total_color += total_color_c1;

                    color_in_01 = __shfl_sync((1U << lotree.sh_data_dim) - 1, color_in_01, lane_colorgrp * lotree.basis_dim);
                    const float grad_common = weight * color_in_01 * gout;
                    const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                    // float beta_grad_0 = -delta_t * ray.delta_scale * _EXP(-pcnt) *
                    //                     (((sdf / (2.0 * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0])) - sigma * lotree.Beta[0]) / (lotree.Beta[0] * lotree.Beta[0]));
                    // float beta_grad_1 = last_beta_grad * _EXP(-pcnt) + beta_grad_0 * _EXP(log_transmit);
                    // float curr_beta_grad = (last_beta_grad - beta_grad_1) * total_color;

                    // last_beta_grad = beta_grad_1;

                    log_transmit -= pcnt;

                    accum -= weight * total_color;
                    float curr_grad_sdf = delta_t * ray.delta_scale * (total_color * _EXP(log_transmit) - accum) *
                                          (-1.0 / (2.0 * lotree.Beta[0] * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0]));

                    trilerp_backward_cuvol_one_LOT(node_corner_ids, grads.grad_sh_out,
                                                   lotree.sh_data_dim,
                                                   pos,
                                                   curr_grad_color, lane_id);
                    if (lane_id == 0)
                    {
                        trilerp_backward_cuvol_one_density_LOT(
                            node_corner_ids,
                            grads.grad_sdf_out,
                            pos, curr_grad_sdf);

                        // atomicAdd(&grads.grad_beta_out[0], curr_beta_grad);
                    }
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        break;
                    }
                }
                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_presample_volsdf_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            float *__restrict__ hit_num,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            const float *__restrict__ presample_t,
            const float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;
            const uint32_t leader_mask = 1U | (1U << lotree.basis_dim) | (1U << (2 * lotree.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            const int tree_N = 2;

            if (ray.tmin > ray.tmax)
            {
                return;
            }

            float t = presample_t[0];

            const float gout = grad_output[lane_colorgrp];

            float log_transmit = 0.f;
            float cube_sz;

            float last_beta_grad = 0.0;

            // remat samples
            float pos[3];
            while (t < presample_t[1])
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                  lotree.CornerSDF,
                                                  1, pos, 0);

                float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                const float delta_t = presample_t[2];

                float lane_color = trilerp_cuvol_one_LOT(
                    node_corner_ids,
                    lotree.CornerSH,
                    lotree.sh_data_dim, pos, lane_id);
                float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                const float pcnt = delta_t * ray.delta_scale * sigma;
                const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));

                const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                float total_color = fmaxf(lane_color_total, 0.f);
                float color_in_01 = total_color == lane_color_total;
                total_color *= gout; // Clamp to [+0, infty)

                float total_color_c1 = __shfl_sync(leader_mask, total_color, lotree.basis_dim);
                total_color += __shfl_sync(leader_mask, total_color, 2 * lotree.basis_dim);
                total_color += total_color_c1;

                color_in_01 = __shfl_sync((1U << lotree.sh_data_dim) - 1, color_in_01, lane_colorgrp * lotree.basis_dim);
                const float grad_common = weight * color_in_01 * gout;
                const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                float beta_grad_0 = -delta_t * ray.delta_scale * _EXP(-pcnt) *
                                    (((sdf / (2.0 * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0])) - sigma * lotree.Beta[0]) / (lotree.Beta[0] * lotree.Beta[0]));
                float beta_grad_1 = last_beta_grad * _EXP(-pcnt) + beta_grad_0 * _EXP(log_transmit);
                float curr_beta_grad = (last_beta_grad - beta_grad_1) * total_color;

                last_beta_grad = beta_grad_1;

                log_transmit -= pcnt;

                accum -= weight * total_color;
                float curr_grad_sdf = delta_t * ray.delta_scale * (total_color * _EXP(log_transmit) - accum) *
                                      (-1.0 / (2.0 * lotree.Beta[0] * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0]));

                trilerp_backward_cuvol_one_LOT(node_corner_ids, grads.grad_sh_out,
                                               lotree.sh_data_dim,
                                               pos,
                                               curr_grad_color, lane_id);
                if (lane_id == 0)
                {
                    trilerp_backward_cuvol_one_density_LOT(
                        node_corner_ids,
                        grads.grad_sdf_out,
                        pos, curr_grad_sdf);

                    atomicAdd(&grads.grad_beta_out[0], curr_beta_grad);

                    hit_num[0] += 1;
                }

                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_gaussian_volsdf_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            const float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;
            const uint32_t leader_mask = 1U | (1U << lotree.basis_dim) | (1U << (2 * lotree.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            const int tree_N = 2;

            if (ray.tmin > ray.tmax)
            {
                return;
            }

            float t = ray.tmin;

            const float gout = grad_output[lane_colorgrp];

            float log_transmit = 0.f;
            float cube_sz;

            // float last_beta_grad = 0.0;

            // remat samples
            float pos[3];
            while (t <= ray.tmax)
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                const float delta_t = 1.0 / (cube_sz * opt.t_step);

                if (cube_sz >= opt.cube_thresh)
                {
                    float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                      lotree.CornerGaussSDF,
                                                      1, pos, 0);

                    float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                    float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, pos, lane_id);
                    float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                    const float pcnt = delta_t * ray.delta_scale * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));

                    const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                    float total_color = fmaxf(lane_color_total, 0.f);
                    float color_in_01 = total_color == lane_color_total;
                    total_color *= gout; // Clamp to [+0, infty)

                    float total_color_c1 = __shfl_sync(leader_mask, total_color, lotree.basis_dim);
                    total_color += __shfl_sync(leader_mask, total_color, 2 * lotree.basis_dim);
                    total_color += total_color_c1;

                    color_in_01 = __shfl_sync((1U << lotree.sh_data_dim) - 1, color_in_01, lane_colorgrp * lotree.basis_dim);
                    const float grad_common = weight * color_in_01 * gout;
                    const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                    // float beta_grad_0 = -delta_t * ray.delta_scale * _EXP(-pcnt) *
                    //                     (((sdf / (2.0 * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0])) - sigma * lotree.Beta[0]) / (lotree.Beta[0] * lotree.Beta[0]));
                    // float beta_grad_1 = last_beta_grad * _EXP(-pcnt) + beta_grad_0 * _EXP(log_transmit);
                    // float curr_beta_grad = (last_beta_grad - beta_grad_1) * total_color;

                    // last_beta_grad = beta_grad_1;

                    log_transmit -= pcnt;

                    accum -= weight * total_color;
                    float curr_grad_sdf = delta_t * ray.delta_scale * (total_color * _EXP(log_transmit) - accum) *
                                          (-1.0 / (2.0 * lotree.Beta[0] * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0]));

                    trilerp_backward_cuvol_one_LOT(node_corner_ids, grads.grad_sh_out,
                                                   lotree.sh_data_dim,
                                                   pos,
                                                   curr_grad_color, lane_id);
                    if (lane_id == 0)
                    {
                        trilerp_backward_cuvol_one_density_LOT(
                            node_corner_ids,
                            grads.grad_sdf_out,
                            pos, curr_grad_sdf);

                        // atomicAdd(&grads.grad_beta_out[0], curr_beta_grad);
                    }
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        break;
                    }
                }
                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_gaussian_presample_volsdf_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            float *__restrict__ hit_num,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            const float *__restrict__ presample_t,
            const float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;
            const uint32_t leader_mask = 1U | (1U << lotree.basis_dim) | (1U << (2 * lotree.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            const int tree_N = 2;

            if (ray.tmin > ray.tmax)
            {
                return;
            }

            float t = presample_t[0];

            const float gout = grad_output[lane_colorgrp];

            float log_transmit = 0.f;
            float cube_sz;

            float last_beta_grad = 0.0;

            // remat samples
            float pos[3];
            while (t < presample_t[1])
            {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                int32_t node_corner_ids[8];
                query_single_from_root_LOT(lotree.child,
                                           lotree.NodeCorners, pos,
                                           &cube_sz, node_corner_ids);

                float sdf = trilerp_cuvol_one_LOT(node_corner_ids,
                                                  lotree.CornerGaussSDF,
                                                  1, pos, 0);

                const float delta_t = presample_t[2];

                float sdf_sign = (sdf >= 0.0) ? 1.0 : -1.0;
                float sigma = (0.5 + 0.5 * sdf_sign * (_EXP(-fabsf(sdf) / lotree.Beta[0]) - 1.0)) / lotree.Beta[0];

                float lane_color = trilerp_cuvol_one_LOT(
                    node_corner_ids,
                    lotree.CornerSH,
                    lotree.sh_data_dim, pos, lane_id);
                float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                const float pcnt = delta_t * ray.delta_scale * sigma;
                const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));

                const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                float total_color = fmaxf(lane_color_total, 0.f);
                float color_in_01 = total_color == lane_color_total;
                total_color *= gout; // Clamp to [+0, infty)

                float total_color_c1 = __shfl_sync(leader_mask, total_color, lotree.basis_dim);
                total_color += __shfl_sync(leader_mask, total_color, 2 * lotree.basis_dim);
                total_color += total_color_c1;

                color_in_01 = __shfl_sync((1U << lotree.sh_data_dim) - 1, color_in_01, lane_colorgrp * lotree.basis_dim);
                const float grad_common = weight * color_in_01 * gout;
                const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                float beta_grad_0 = -delta_t * ray.delta_scale * _EXP(-pcnt) *
                                    (((sdf / (2.0 * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0])) - sigma * lotree.Beta[0]) / (lotree.Beta[0] * lotree.Beta[0]));
                float beta_grad_1 = last_beta_grad * _EXP(-pcnt) + beta_grad_0 * _EXP(log_transmit);
                float curr_beta_grad = (last_beta_grad - beta_grad_1) * total_color;

                last_beta_grad = beta_grad_1;

                log_transmit -= pcnt;

                accum -= weight * total_color;
                float curr_grad_sdf = delta_t * ray.delta_scale * (total_color * _EXP(log_transmit) - accum) *
                                      (-1.0 / (2.0 * lotree.Beta[0] * lotree.Beta[0]) * _EXP(-fabsf(sdf) / lotree.Beta[0]));

                trilerp_backward_cuvol_one_LOT(node_corner_ids, grads.grad_sh_out,
                                               lotree.sh_data_dim,
                                               pos,
                                               curr_grad_color, lane_id);
                if (lane_id == 0)
                {
                    trilerp_backward_cuvol_one_density_LOT(
                        node_corner_ids,
                        grads.grad_sdf_out,
                        pos, curr_grad_sdf);

                    atomicAdd(&grads.grad_beta_out[0], curr_beta_grad);

                    hit_num[0] += 1;
                }

                t += delta_t;
            }
        }

        __device__ __inline__ void trace_ray_sdf_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            SingleRayHitLOTreeSDF &__restrict__ ray_hls,
            const float *__restrict__ grad_output,
            const float *__restrict__ color_cache,
            const RenderOptions &__restrict__ opt,
            uint32_t lane_id,
            const float *__restrict__ sphfunc_val,
            float *__restrict__ grad_sphfunc_val,
            WarpReducef::TempStorage &__restrict__ temp_storage,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            const uint32_t lane_colorgrp_id = lane_id % lotree.basis_dim;
            const uint32_t lane_colorgrp = lane_id / lotree.basis_dim;
            const uint32_t leader_mask = 1U | (1U << lotree.basis_dim) | (1U << (2 * lotree.basis_dim));

            float accum = fmaf(color_cache[0], grad_output[0],
                               fmaf(color_cache[1], grad_output[1],
                                    color_cache[2] * grad_output[2]));

            const int tree_N = 2;

            if (ray_hls.hitnum <= 1)
            {
                return;
            }

            const float gout = grad_output[lane_colorgrp];

            float sdf_p = 0.0;
            float sdf_l = 0.0;

            float accum_trans = 1.0;

            float last_accum_trans = 1.0;
            float last_alpha = 0.0;
            float last_total_color = 0.0;
            float last_alpha_s = 1.0;
            float last_s_grad_accum = 0.0;
            float last_s_grad_single = 0.0;

            float c_sdf_point_p[3];
            for (int j = 0; j < 3; ++j)
            {
                c_sdf_point_p[j] = ray_hls.sdf_point[j];
            }

            int32_t node_corner_ids_p[8];
            for (int j = 0; j < 8; ++j)
            {
                node_corner_ids_p[j] = ray_hls.hitnode_sdf[j];
            }

            sdf_p = trilerp_cuvol_one_LOT(node_corner_ids_p,
                                          lotree.CornerSDF,
                                          1, c_sdf_point_p, 0);

            for (int on = 0; on < ray_hls.hitnum - 1; ++on)
            {
                float c_sdf_point_n[3];
                for (int j = 0; j < 3; ++j)
                {
                    c_sdf_point_n[j] = ray_hls.sdf_point[(on + 1) * 3 + j];
                }

                int32_t node_corner_ids_n[8];
                for (int j = 0; j < 8; ++j)
                {
                    node_corner_ids_n[j] = ray_hls.hitnode_sdf[(on + 1) * 8 + j];
                }

                float sdf_n = trilerp_cuvol_one_LOT(node_corner_ids_n,
                                                    lotree.CornerSDF,
                                                    1, c_sdf_point_n, 0);

                float cdf_p = 1.0 / (1.0 + expf(-lotree.LearnS[0] * sdf_p));
                float cdf_n = 1.0 / (1.0 + expf(-lotree.LearnS[0] * sdf_n));

                float alpha = min(max((cdf_p - cdf_n + 1e-5) / (cdf_p + 1e-5), 0.0), 1.0);

                if (alpha > opt.alpha_thresh)
                {
                    float c_col_point[3];
                    for (int j = 0; j < 3; ++j)
                    {
                        c_col_point[j] = ray_hls.col_point[on * 3 + j];
                    }

                    int32_t node_corner_ids[8];
                    for (int j = 0; j < 8; ++j)
                    {
                        node_corner_ids[j] = ray_hls.hitnode_col[on * 8 + j];
                    }

                    float lane_color = trilerp_cuvol_one_LOT(
                        node_corner_ids,
                        lotree.CornerSH,
                        lotree.sh_data_dim, c_col_point, lane_id);
                    float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

                    const float weight = accum_trans * alpha;

                    const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
                    float total_color = fmaxf(lane_color_total, 0.f);
                    float color_in_01 = total_color == lane_color_total;
                    total_color *= gout; // Clamp to [+0, infty)

                    float total_color_c1 = __shfl_sync(leader_mask, total_color, lotree.basis_dim);
                    total_color += __shfl_sync(leader_mask, total_color, 2 * lotree.basis_dim);
                    total_color += total_color_c1;

                    color_in_01 = __shfl_sync((1U << lotree.sh_data_dim) - 1, color_in_01, lane_colorgrp * lotree.basis_dim);
                    const float grad_common = weight * color_in_01 * gout;
                    const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

                    float diff_nsdfv = -999;
                    float diff_psdfv = -999;

                    float alpha_grad_0 = 0.0;
                    if (last_alpha != 0.0)
                    {
                        diff_nsdfv = diff_nsdf(sdf_l, sdf_p, lotree.LearnS[0]);
                        alpha_grad_0 = last_accum_trans * last_total_color * diff_nsdfv;
                    }

                    float alpha_grad_1 = 0.0;
                    if (last_alpha != 0.0)
                    {
                        if (diff_nsdfv == -999)
                        {
                            diff_nsdfv = diff_nsdf(sdf_l, sdf_p, lotree.LearnS[0]);
                        }

                        if (diff_psdfv == -999)
                        {
                            diff_psdfv = diff_psdf(sdf_p, sdf_n, lotree.LearnS[0]);
                        }

                        alpha_grad_1 = last_accum_trans * total_color *
                                       (diff_psdfv - diff_nsdfv * alpha - last_alpha * diff_psdfv);
                    }
                    else
                    {
                        if (diff_psdfv == -999)
                        {
                            diff_psdfv = diff_psdf(sdf_p, sdf_n, lotree.LearnS[0]);
                        }

                        alpha_grad_1 = last_accum_trans * total_color * diff_psdfv;
                    }

                    float weight_color = weight * total_color;
                    accum -= weight_color;

                    float alpha_grad_2 = 0.0;
                    if ((accum != 0.0) && (on != ray_hls.hitnum - 2))
                    {
                        if (last_alpha != 0.0 && last_alpha != 1.0 && alpha != 1.0)
                        {
                            if (diff_nsdfv == -999)
                            {
                                diff_nsdfv = diff_nsdf(sdf_l, sdf_p, lotree.LearnS[0]);
                            }

                            if (diff_psdfv == -999)
                            {
                                diff_psdfv = diff_psdf(sdf_p, sdf_n, lotree.LearnS[0]);
                            }

                            alpha_grad_2 = (accum / ((1.0 - last_alpha) * (1.0 - alpha))) *
                                           (-diff_nsdfv - diff_psdfv +
                                            diff_nsdfv * alpha + last_alpha * diff_psdfv);
                        }
                        else if (alpha != 1.0)
                        {
                            if (diff_psdfv == -999)
                            {
                                diff_psdfv = diff_psdf(sdf_p, sdf_n, lotree.LearnS[0]);
                            }

                            alpha_grad_2 = (accum / (1.0 - alpha)) * (-diff_psdfv);
                        }
                    }

                    float curr_grad_alpha = alpha_grad_0 + alpha_grad_1 + alpha_grad_2;

                    float s_grad_0 = last_s_grad_accum * (1.0 - last_alpha_s);
                    float s_grad_1 = -last_s_grad_single;
                    float s_grad_01 = (s_grad_0 + s_grad_1) * alpha;

                    float c_s_diff = diff_ssdf(sdf_p, sdf_n, lotree.LearnS[0]);
                    float s_grad_2 = accum_trans * c_s_diff;

                    float curr_s_grad = (s_grad_01 + s_grad_2) * total_color;

                    last_s_grad_accum = s_grad_0 + s_grad_1;
                    last_s_grad_single = s_grad_2;
                    last_alpha_s = alpha;

                    last_total_color = total_color;
                    last_accum_trans = accum_trans;
                    last_alpha = alpha;

                    trilerp_backward_cuvol_one_LOT(node_corner_ids, grads.grad_sh_out,
                                                   lotree.sh_data_dim,
                                                   c_col_point,
                                                   curr_grad_color, lane_id);

                    if (lane_id == 0)
                    {
                        float c_sdf_point_p[3];
                        for (int j = 0; j < 3; ++j)
                        {
                            c_sdf_point_p[j] = ray_hls.sdf_point[on * 3 + j];
                        }

                        int32_t node_corner_ids_p[8];
                        for (int j = 0; j < 8; ++j)
                        {
                            node_corner_ids_p[j] = ray_hls.hitnode_sdf[on * 8 + j];
                        }

                        trilerp_backward_cuvol_one_density_LOT(
                            node_corner_ids_p,
                            grads.grad_sdf_out,
                            c_sdf_point_p, curr_grad_alpha);

                        if (on == ray_hls.hitnum - 2)
                        {
                            diff_nsdfv = diff_nsdf(sdf_p, sdf_n, lotree.LearnS[0]);
                            alpha_grad_2 = accum_trans * total_color * diff_nsdfv;

                            trilerp_backward_cuvol_one_density_LOT(
                                node_corner_ids_n,
                                grads.grad_sdf_out,
                                c_sdf_point_n, alpha_grad_2);
                        }

                        atomicAdd(&grads.grad_learns_out[0], curr_s_grad);
                    }

                    accum_trans *= (1.0 - alpha + 1e-7);

                    if (accum_trans < opt.stop_thresh)
                    {
                        break;
                    }
                }
                else
                {
                    if (last_alpha != 0.0)
                    {
                        float diff_nsdfv = diff_nsdf(sdf_l, sdf_p, lotree.LearnS[0]);

                        float alpha_grad_0 = 0.0;
                        if (last_accum_trans != 0.0)
                        {
                            alpha_grad_0 = last_accum_trans * last_total_color * diff_nsdfv;
                        }

                        float alpha_grad_2 = 0.0;
                        if ((accum != 0.0) && (on != ray_hls.hitnum - 2) && (last_alpha != 1.0))
                        {
                            alpha_grad_2 = (accum / (1.0 - last_alpha)) * (-diff_nsdfv);
                        }

                        float curr_grad_alpha = alpha_grad_0 + alpha_grad_2;

                        if (lane_id == 0)
                        {
                            float c_sdf_point_p[3];
                            for (int j = 0; j < 3; ++j)
                            {
                                c_sdf_point_p[j] = ray_hls.sdf_point[on * 3 + j];
                            }

                            int32_t node_corner_ids_p[8];
                            for (int j = 0; j < 8; ++j)
                            {
                                node_corner_ids_p[j] = ray_hls.hitnode_sdf[on * 8 + j];
                            }

                            trilerp_backward_cuvol_one_density_LOT(
                                node_corner_ids_p,
                                grads.grad_sdf_out,
                                c_sdf_point_p, curr_grad_alpha);
                        }
                    }

                    last_alpha = 0.0;
                    last_total_color = 0.0;
                    last_accum_trans = accum_trans;
                }

                sdf_p = sdf_n;
                sdf_l = sdf_p;
            }
        }

        __device__ __inline__ void trace_ray_eikonal_backward_LOT(
            const PackedTreeSpecLOT &__restrict__ lotree,
            float *__restrict__ sdf_point,
            int *__restrict__ hitnode_sdf,
            int hitnum,
            const float *__restrict__ sdf_grad,
            const float scale_grad,
            PackedGridOutputGradsSDFLOT &__restrict__ grads)
        {
            for (int on = 0; on < hitnum; ++on)
            {
                float c_sdf_grad_x2 = sdf_grad[on * 3] * sdf_grad[on * 3];
                float c_sdf_grad_y2 = sdf_grad[on * 3 + 1] * sdf_grad[on * 3 + 1];
                float c_sdf_grad_z2 = sdf_grad[on * 3 + 2] * sdf_grad[on * 3 + 2];

                float c_sdf_grad_norm = sqrt(c_sdf_grad_x2 + c_sdf_grad_y2 + c_sdf_grad_z2);

                float loss_grad_0 = c_sdf_grad_norm - 1.0;
                float loss_grad_1 = 1.0 / (c_sdf_grad_norm + 1e-8);

                float loss_grad_01 = scale_grad * loss_grad_0 * loss_grad_1;

                float node_corner_grad[8];
                float c_sdf_point[3];
                for (int j = 0; j < 3; ++j)
                {
                    c_sdf_point[j] = sdf_point[on * 3 + j];
                }

                /********************************************/

                float loss_grad_2_x = 2.0 * sdf_grad[on * 3];
                float loss_grad_012_x = loss_grad_01 * loss_grad_2_x;
                trilerp_backward_cuvol_one_eikonal_LOT(node_corner_grad, c_sdf_point, loss_grad_012_x);

                for (int j = 0; j < 8; ++j)
                {
                    int32_t node_corner_id = hitnode_sdf[on * 8 + j];

                    int32_t node_corner_x0 = lotree.NodeAllNeighbors[node_corner_id * 6];
                    int32_t node_corner_x1 = lotree.NodeAllNeighbors[node_corner_id * 6 + 1];

                    if (node_corner_x0 != -1 && node_corner_x1 != -1)
                    {
                        float x0_length = lotree.NodeAllNeighLen[node_corner_id * 6];
                        float x1_length = lotree.NodeAllNeighLen[node_corner_id * 6 + 1];

                        float total_lenth = x0_length + x1_length;

                        float loss_grad_3_0_x = -x1_length / (total_lenth * x0_length);
                        atomicAdd(&grads.grad_sdf_out[node_corner_x0], loss_grad_3_0_x * node_corner_grad[j]);

                        float loss_grad_3_1_x = x0_length / (total_lenth * x1_length);
                        atomicAdd(&grads.grad_sdf_out[node_corner_x1], loss_grad_3_1_x * node_corner_grad[j]);

                        float loss_grad_3_x = (-x0_length / (total_lenth * x1_length)) + (x1_length / (total_lenth * x0_length));
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], loss_grad_3_x * node_corner_grad[j]);
                    }
                    else if (node_corner_x0 != -1)
                    {
                        float x0_length_inv = 1.0 / lotree.NodeAllNeighLen[node_corner_id * 6];

                        atomicAdd(&grads.grad_sdf_out[node_corner_x0], -x0_length_inv * node_corner_grad[j]);
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], x0_length_inv * node_corner_grad[j]);
                    }
                    else if (node_corner_x1 != -1)
                    {
                        float x1_length_inv = 1.0 / lotree.NodeAllNeighLen[node_corner_id * 6 + 1];

                        atomicAdd(&grads.grad_sdf_out[node_corner_x1], x1_length_inv * node_corner_grad[j]);
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], -x1_length_inv * node_corner_grad[j]);
                    }
                }

                /********************************************/

                float loss_grad_2_y = 2.0 * sdf_grad[on * 3 + 1];
                float loss_grad_012_y = loss_grad_01 * loss_grad_2_y;
                trilerp_backward_cuvol_one_eikonal_LOT(node_corner_grad, c_sdf_point, loss_grad_012_y);

                for (int j = 0; j < 8; ++j)
                {
                    int32_t node_corner_id = hitnode_sdf[on * 8 + j];
                    int32_t node_corner_y0 = lotree.NodeAllNeighbors[node_corner_id * 6 + 2];
                    int32_t node_corner_y1 = lotree.NodeAllNeighbors[node_corner_id * 6 + 3];

                    if (node_corner_y0 != -1 && node_corner_y1 != -1)
                    {
                        float y0_length = lotree.NodeAllNeighLen[node_corner_id * 6 + 2];
                        float y1_length = lotree.NodeAllNeighLen[node_corner_id * 6 + 3];

                        float total_lenth = y0_length + y1_length;

                        float loss_grad_3_0_y = -y1_length / (total_lenth * y0_length);
                        atomicAdd(&grads.grad_sdf_out[node_corner_y0], loss_grad_3_0_y * node_corner_grad[j]);

                        float loss_grad_3_1_y = y0_length / (total_lenth * y1_length);
                        atomicAdd(&grads.grad_sdf_out[node_corner_y1], loss_grad_3_1_y * node_corner_grad[j]);

                        float loss_grad_3_y = (-y0_length / (total_lenth * y1_length)) + (y1_length / (total_lenth * y0_length));
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], loss_grad_3_y * node_corner_grad[j]);
                    }
                    else if (node_corner_y0 != -1)
                    {
                        float y0_length_inv = 1.0 / lotree.NodeAllNeighLen[node_corner_id * 6 + 2];

                        atomicAdd(&grads.grad_sdf_out[node_corner_y0], -y0_length_inv * node_corner_grad[j]);
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], y0_length_inv * node_corner_grad[j]);
                    }
                    else if (node_corner_y1 != -1)
                    {
                        float y1_length_inv = 1.0 / lotree.NodeAllNeighLen[node_corner_id * 6 + 3];

                        atomicAdd(&grads.grad_sdf_out[node_corner_y1], y1_length_inv * node_corner_grad[j]);
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], -y1_length_inv * node_corner_grad[j]);
                    }
                }

                /********************************************/

                float loss_grad_2_z = 2.0 * sdf_grad[on * 3 + 2];
                float loss_grad_012_z = loss_grad_01 * loss_grad_2_z;
                trilerp_backward_cuvol_one_eikonal_LOT(node_corner_grad, c_sdf_point, loss_grad_012_z);

                for (int j = 0; j < 8; ++j)
                {
                    int32_t node_corner_id = hitnode_sdf[on * 8 + j];
                    int32_t node_corner_z0 = lotree.NodeAllNeighbors[node_corner_id * 6 + 4];
                    int32_t node_corner_z1 = lotree.NodeAllNeighbors[node_corner_id * 6 + 5];

                    if (node_corner_z0 != -1 && node_corner_z1 != -1)
                    {
                        float z0_length = lotree.NodeAllNeighLen[node_corner_id * 6 + 4];
                        float z1_length = lotree.NodeAllNeighLen[node_corner_id * 6 + 5];

                        float total_lenth = z0_length + z1_length;

                        float loss_grad_3_0_z = -z1_length / (total_lenth * z0_length);
                        atomicAdd(&grads.grad_sdf_out[node_corner_z0], loss_grad_3_0_z * node_corner_grad[j]);

                        float loss_grad_3_1_z = z0_length / (total_lenth * z1_length);
                        atomicAdd(&grads.grad_sdf_out[node_corner_z1], loss_grad_3_1_z * node_corner_grad[j]);

                        float loss_grad_3_z = (-z0_length / (total_lenth * z1_length)) + (z1_length / (total_lenth * z0_length));
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], loss_grad_3_z * node_corner_grad[j]);
                    }
                    else if (node_corner_z0 != -1)
                    {
                        float z0_length_inv = 1.0 / lotree.NodeAllNeighLen[node_corner_id * 6 + 4];

                        atomicAdd(&grads.grad_sdf_out[node_corner_z0], -z0_length_inv * node_corner_grad[j]);
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], z0_length_inv * node_corner_grad[j]);
                    }
                    else if (node_corner_z1 != -1)
                    {
                        float z1_length_inv = 1.0 / lotree.NodeAllNeighLen[node_corner_id * 6 + 5];

                        atomicAdd(&grads.grad_sdf_out[node_corner_z1], z1_length_inv * node_corner_grad[j]);
                        atomicAdd(&grads.grad_sdf_out[node_corner_id], -z1_length_inv * node_corner_grad[j]);
                    }
                }
            }
        }

        __device__ __inline__ void render_background_forward(
            const PackedSparseGridSpec &__restrict__ grid,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            float log_transmit,
            float *__restrict__ out)
        {

            ConcentricSpheresIntersector csi(ray.origin, ray.dir);

            const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
            float t, invr_last = 1.f / inner_radius;
            const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

            // csi.intersect(inner_radius, &t_last);

            float outv[3] = {0.f, 0.f, 0.f};
            for (int i = 0; i < n_steps; ++i)
            {
                // Between 1 and infty
                float r = n_steps / (n_steps - i - 0.5);
                if (r < inner_radius || !csi.intersect(r, &t))
                    continue;

#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }
                const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] *= invr_mid;
                }
                // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
                _unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
                ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                                   grid.background_nlayers - 1);
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.l[j] = (int)ray.pos[j];
                }
                ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
                ray.l[1] = min(ray.l[1], grid.background_reso - 1);
                ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] -= ray.l[j];
                }

                float sigma = trilerp_bg_one(
                    grid.background_links,
                    grid.background_data,
                    grid.background_reso,
                    grid.background_nlayers,
                    4,
                    ray.l,
                    ray.pos,
                    3);

                // if (i == n_steps - 1) {
                //     ray.world_step = 1e9;
                // }
                // if (opt.randomize && opt.random_sigma_std_background > 0.0)
                //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
                if (sigma > 0.f)
                {
                    const float pcnt = (invr_last - invr_mid) * ray.world_step * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;
#pragma unroll 3
                    for (int i = 0; i < 3; ++i)
                    {
                        // Not efficient
                        const float color = trilerp_bg_one(
                                                grid.background_links,
                                                grid.background_data,
                                                grid.background_reso,
                                                grid.background_nlayers,
                                                4,
                                                ray.l,
                                                ray.pos,
                                                i) *
                                            C0;                       // Scale by SH DC factor to help normalize lrs
                        outv[i] += weight * fmaxf(color + 0.5f, 0.f); // Clamp to [+0, infty)
                    }
                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        break;
                    }
                }
                invr_last = invr_mid;
            }
#pragma unroll 3
            for (int i = 0; i < 3; ++i)
            {
                out[i] += outv[i] + _EXP(log_transmit) * opt.background_brightness;
            }
        }

        __device__ __inline__ void render_background_backward(
            const PackedSparseGridSpec &__restrict__ grid,
            const float *__restrict__ grad_output,
            SingleRaySpec &__restrict__ ray,
            const RenderOptions &__restrict__ opt,
            float log_transmit,
            float accum,
            float sparsity_loss,
            PackedGridOutputGrads &__restrict__ grads)
        {
            // printf("accum_init=%f\n", accum);
            // printf("log_transmit_init=%f\n", log_transmit);
            ConcentricSpheresIntersector csi(ray.origin, ray.dir);

            const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

            const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
            float t, invr_last = 1.f / inner_radius;
            // csi.intersect(inner_radius, &t_last);
            for (int i = 0; i < n_steps; ++i)
            {
                float r = n_steps / (n_steps - i - 0.5);

                if (r < inner_radius || !csi.intersect(r, &t))
                    continue;

#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
                }

                const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] *= invr_mid;
                }
                // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
                _unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
                ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                                   grid.background_nlayers - 1);
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.l[j] = (int)ray.pos[j];
                }
                ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
                ray.l[1] = min(ray.l[1], grid.background_reso - 1);
                ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
                for (int j = 0; j < 3; ++j)
                {
                    ray.pos[j] -= ray.l[j];
                }

                float sigma = trilerp_bg_one(
                    grid.background_links,
                    grid.background_data,
                    grid.background_reso,
                    grid.background_nlayers,
                    4,
                    ray.l,
                    ray.pos,
                    3);
                // if (i == n_steps - 1) {
                //     ray.world_step = 1e9;
                // }

                // if (opt.randomize && opt.random_sigma_std_background > 0.0)
                //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
                if (sigma > 0.f)
                {
                    float total_color = 0.f;
                    const float pcnt = ray.world_step * (invr_last - invr_mid) * sigma;
                    const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
                    log_transmit -= pcnt;

                    for (int i = 0; i < 3; ++i)
                    {
                        const float color = trilerp_bg_one(
                                                grid.background_links,
                                                grid.background_data,
                                                grid.background_reso,
                                                grid.background_nlayers,
                                                4,
                                                ray.l,
                                                ray.pos,
                                                i) *
                                                C0 +
                                            0.5f; // Scale by SH DC factor to help normalize lrs

                        total_color += fmaxf(color, 0.f) * grad_output[i];
                        if (color > 0.f)
                        {
                            const float curr_grad_color = C0 * weight * grad_output[i];
                            trilerp_backward_bg_one(
                                grid.background_links,
                                grads.grad_background_out,
                                nullptr,
                                grid.background_reso,
                                grid.background_nlayers,
                                4,
                                ray.l,
                                ray.pos,
                                curr_grad_color,
                                i);
                        }
                    }

                    accum -= weight * total_color;
                    float curr_grad_sigma = ray.world_step * (invr_last - invr_mid) * (total_color * _EXP(log_transmit) - accum);
                    if (sparsity_loss > 0.f)
                    {
                        // Cauchy version (from SNeRG)
                        curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                        // Alphs version (from PlenOctrees)
                        // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
                    }

                    trilerp_backward_bg_one(
                        grid.background_links,
                        grads.grad_background_out,
                        grads.mask_background_out,
                        grid.background_reso,
                        grid.background_nlayers,
                        4,
                        ray.l,
                        ray.pos,
                        curr_grad_sigma,
                        3);

                    if (_EXP(log_transmit) < opt.stop_thresh)
                    {
                        break;
                    }
                }
                invr_last = invr_mid;
            }
        }

        // BEGIN KERNELS

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_kernel(
                PackedSparseGridSpec grid,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
                float *__restrict__ log_transmit_out = nullptr)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= grid.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            calc_sphfunc(grid, lane_id,
                         ray_id,
                         ray_spec[ray_blk_id].dir,
                         sphfunc_val[ray_blk_id]);
            ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
            __syncwarp((1U << grid.sh_data_dim) - 1);

            trace_ray_cuvol(
                grid,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data(),
                log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
                float *__restrict__ log_transmit_out = nullptr)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_cuvol_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data(),
                log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_volsdf_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_convert_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_volsdf_convert_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_downsample_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<int32_t, 5, torch::RestrictPtrTraits> leaf_node_map,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_volsdf_downsample_test_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                leaf_node_map.data(),
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_downsample_render_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_volsdf_downsample_render_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_presample_volsdf_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> presample_t,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_presample_volsdf_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                presample_t[ray_id].data(),
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_gaussian_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_gaussian_volsdf_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_presample_volsdf_gaussian_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> presample_t,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_gaussian_presample_volsdf_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                presample_t[ray_id].data(),
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_record_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_volsdf_record_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_record_w_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_volsdf_record_w_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_presample_volsdf_record_w_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> presample,
                torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_presample_volsdf_record_w_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                presample[ray_id].data(),
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_sdf_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                PackedRaysHitLOTreeSDF rays_hls,
                int rays_hls_stride,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ SingleRayHitLOTreeSDF ray_hls[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            ray_hls[ray_blk_id].set(rays_hls.sdf_point + ray_id * rays_hls_stride * 3,
                                    rays_hls.col_point + ray_id * rays_hls_stride * 3,
                                    rays_hls.hitnode_sdf + ray_id * rays_hls_stride * 8,
                                    rays_hls.hitnode_col + ray_id * rays_hls_stride * 8,
                                    rays_hls.hitnum[ray_id]);

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_sdf_LOT(
                lotree,
                ray_hls[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out[ray_id].data());
        }

        __global__ void render_ray_volsdf_eikonal_kernel_LOT(
            PackedTreeSpecLOT lotree,
            PackedRaysSpec rays,
            RenderOptions opt,
            float scale,
            float *__restrict__ uni_samples,
            float *__restrict__ total_eikonal_loss,
            PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)));

            SingleRaySpec ray_spec;
            ray_spec.set(rays.origins[tid].data(), rays.dirs[tid].data());

            ray_find_bounds_LOT(ray_spec, lotree);

            trace_ray_volsdf_eikonal_LOT(
                lotree,
                ray_spec,
                opt,
                scale,
                uni_samples + tid * 3,
                total_eikonal_loss + tid,
                grads);
        }

        __global__ void render_ray_eikonal_kernel_LOT(
            PackedRaysHitLOTreeSDF rays_hls,
            int rays_hls_stride,
            int rays_size,
            torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> corner_sdf_grad,
            torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> sdf_grad_out)
        {
            CUDA_GET_THREAD_ID(tid, rays_size);

            trace_ray_sdf_grad_LOT(
                corner_sdf_grad.data(),
                rays_hls.sdf_point + tid * rays_hls_stride * 3,
                rays_hls.hitnode_sdf + tid * rays_hls_stride * 8,
                rays_hls.hitnum[tid],
                sdf_grad_out[tid].data());
        }

        __global__ void render_ray_hitnum_kernel_LOT(
            PackedTreeSpecLOT lotree,
            PackedRaysSpec rays,
            RenderOptions opt,
            torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)));

            SingleRaySpec ray_spec;
            ray_spec.set(rays.origins[tid].data(), rays.dirs[tid].data());

            ray_find_bounds_LOT(ray_spec, lotree);

            trace_ray_hitnum_LOT(
                lotree,
                ray_spec,
                opt,
                out[tid].data());
        }

        __global__ void render_ray_volsdf_hitnum_kernel_LOT(
            PackedTreeSpecLOT lotree,
            PackedRaysSpec rays,
            RenderOptions opt,
            torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)));

            SingleRaySpec ray_spec;
            ray_spec.set(rays.origins[tid].data(), rays.dirs[tid].data());

            ray_find_bounds_LOT(ray_spec, lotree);

            trace_ray_volsdf_hitnum_LOT(
                lotree,
                ray_spec,
                opt,
                out[tid].data());
        }

        __global__ void render_ray_volsdf_colrefine_kernel_LOT(
            PackedTreeSpecLOT lotree,
            PackedRaysSpec rays,
            RenderOptions opt,
            torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> ray_mse,
            torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)));

            SingleRaySpec ray_spec;
            ray_spec.set(rays.origins[tid].data(), rays.dirs[tid].data());

            ray_find_bounds_LOT(ray_spec, lotree);

            trace_ray_volsdf_colrefine_LOT(
                lotree,
                ray_spec,
                ray_mse[tid],
                opt,
                out.data());
        }

        __global__ void render_ray_volsdf_prehit_kernel_LOT(
            PackedTreeSpecLOT lotree,
            PackedRaysSpec rays,
            RenderOptions opt,
            torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)));

            SingleRaySpec ray_spec;
            ray_spec.set(rays.origins[tid].data(), rays.dirs[tid].data());

            ray_find_bounds_LOT(ray_spec, lotree);

            trace_ray_volsdf_prehit_LOT(
                lotree,
                ray_spec,
                opt,
                out[tid].data());
        }

        __global__ void render_ray_hitpoint_sdf_kernel_LOT(
            PackedTreeSpecLOT lotree,
            PackedRaysSpec rays,
            float step,
            torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> sdf_point_out,
            torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> col_point_out,
            torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> hitnode_sdf_out,
            torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> hitnode_col_out,
            torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> hitnum_out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)));

            SingleRaySpec ray_spec;
            ray_spec.set(rays.origins[tid].data(), rays.dirs[tid].data());

            ray_find_bounds_LOT(ray_spec, lotree);

            trace_ray_hitpoint_sdf_LOT(
                lotree,
                ray_spec,
                step,
                sdf_point_out[tid].data(),
                col_point_out[tid].data(),
                hitnode_sdf_out[tid].data(),
                hitnode_col_out[tid].data(),
                hitnum_out[tid].data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_refine_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> color_out,
                torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> hitnode_out,
                torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> weight_out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_refine_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                color_out[ray_id].data(),
                hitnode_out[ray_id].data(),
                weight_out.data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_refine_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> color_out,
                torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> hitnode_out,
                torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> weight_out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_volsdf_refine_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                color_out[ray_id].data(),
                hitnode_out[ray_id].data(),
                weight_out.data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_refine_sdf_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysSpec rays,
                RenderOptions opt,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> color_out,
                torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> hitnode_out,
                torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> sdf_out)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim) // Bad, but currently the best way due to coalesced memory access
                return;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());

            calc_sphfunc_LOT(lotree,
                             ray_spec[ray_blk_id].dir,
                             sphfunc_val[ray_blk_id]);

            ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);

            __syncwarp((1U << lotree.sh_data_dim) - 1);

            trace_ray_volsdf_refine_sdf_LOT(
                lotree,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                color_out[ray_id].data(),
                hitnode_out[ray_id].data(),
                sdf_out.data());
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_image_kernel(
                PackedSparseGridSpec grid,
                PackedCameraSpec cam,
                RenderOptions opt,
                float *__restrict__ out,
                float *__restrict__ log_transmit_out = nullptr)
        {
            CUDA_GET_THREAD_ID(tid, cam.height * cam.width * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= grid.sh_data_dim)
                return;

            const int ix = ray_id % cam.width;
            const int iy = ray_id / cam.width;

            __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];

            cam2world_ray(ix, iy, cam, ray_spec[ray_blk_id].dir, ray_spec[ray_blk_id].origin);
            calc_sphfunc(grid, lane_id,
                         ray_id,
                         ray_spec[ray_blk_id].dir,
                         sphfunc_val[ray_blk_id]);
            ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
            __syncwarp((1U << grid.sh_data_dim) - 1);

            trace_ray_cuvol(
                grid,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                out + ray_id * 3,
                log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
        }

        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_backward_kernel(
                PackedSparseGridSpec grid,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                const float *__restrict__ log_transmit_in,
                float beta_loss,
                float sparsity_loss,
                PackedGridOutputGrads grads,
                float *__restrict__ accum_out = nullptr,
                float *__restrict__ log_transmit_out = nullptr)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= grid.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < grid.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc(grid, lane_id,
                         ray_id,
                         vdir, sphfunc_val[ray_blk_id]);
            if (lane_id == 0)
            {
                ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
            }

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << grid.sh_data_dim) - 1);
            trace_ray_cuvol_backward(
                grid,
                grad_out,
                color_cache + ray_id * 3,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
                beta_loss,
                sparsity_loss,
                grads,
                accum_out == nullptr ? nullptr : accum_out + ray_id,
                log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
            calc_sphfunc_backward(
                grid, lane_id,
                ray_id,
                vdir,
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                grads.grad_basis_out);
        }

        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_backward_kernel_LOT(
                PackedTreeSpecLOT lotree,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                const float *__restrict__ log_transmit_in,
                float beta_loss,
                float sparsity_loss,
                PackedGridOutputGradsLOT grads,
                float *__restrict__ accum_out = nullptr,
                float *__restrict__ log_transmit_out = nullptr)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < lotree.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc_LOT(lotree, vdir, sphfunc_val[ray_blk_id]);
            if (lane_id == 0)
            {
                ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);
            }

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << lotree.sh_data_dim) - 1);
            trace_ray_cuvol_backward_LOT(
                lotree,
                grad_out,
                color_cache + ray_id * 3,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
                beta_loss,
                sparsity_loss,
                grads,
                accum_out == nullptr ? nullptr : accum_out + ray_id,
                log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
        }

        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_backward_kernel_LOT(
                PackedTreeSpecLOT lotree,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < lotree.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc_LOT(lotree, vdir, sphfunc_val[ray_blk_id]);
            if (lane_id == 0)
            {
                ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);
            }

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << lotree.sh_data_dim) - 1);
            trace_ray_volsdf_backward_LOT(
                lotree,
                grad_out,
                color_cache + ray_id * 3,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                grads);
        }

        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_convert_backward_kernel_LOT(
                PackedTreeSpecLOT lotree,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < lotree.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc_LOT(lotree, vdir, sphfunc_val[ray_blk_id]);
            if (lane_id == 0)
            {
                ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);
            }

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << lotree.sh_data_dim) - 1);
            trace_ray_volsdf_convert_backward_LOT(
                lotree,
                grad_out,
                color_cache + ray_id * 3,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                grads);
        }

        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_downsample_backward_kernel_LOT(
                PackedTreeSpecLOT lotree,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                const int32_t *__restrict__ leaf_node_map,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < lotree.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc_LOT(lotree, vdir, sphfunc_val[ray_blk_id]);
            if (lane_id == 0)
            {
                ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);
            }

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << lotree.sh_data_dim) - 1);
            trace_ray_volsdf_downsample_backward_LOT(
                lotree,
                grad_out,
                color_cache + ray_id * 3,
                leaf_node_map,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                grads);
        }


        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_presample_volsdf_backward_kernel_LOT(
                PackedTreeSpecLOT lotree,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                float *__restrict__ hit_num,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> presample_t,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < lotree.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc_LOT(lotree, vdir, sphfunc_val[ray_blk_id]);
            if (lane_id == 0)
            {
                ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);
            }

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << lotree.sh_data_dim) - 1);
            trace_ray_presample_volsdf_backward_LOT(
                lotree,
                grad_out,
                color_cache + ray_id * 3,
                hit_num + ray_id,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                presample_t[ray_id].data(),
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                grads);
        }

        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_volsdf_gaussian_backward_kernel_LOT(
                PackedTreeSpecLOT lotree,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < lotree.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc_LOT(lotree, vdir, sphfunc_val[ray_blk_id]);
            if (lane_id == 0)
            {
                ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);
            }

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << lotree.sh_data_dim) - 1);
            trace_ray_gaussian_volsdf_backward_LOT(
                lotree,
                grad_out,
                color_cache + ray_id * 3,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                grads);
        }

        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_presample_volsdf_gaussian_backward_kernel_LOT(
                PackedTreeSpecLOT lotree,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                float *__restrict__ hit_num,
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> presample_t,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < lotree.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc_LOT(lotree, vdir, sphfunc_val[ray_blk_id]);
            if (lane_id == 0)
            {
                ray_find_bounds_LOT(ray_spec[ray_blk_id], lotree);
            }

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << lotree.sh_data_dim) - 1);
            trace_ray_gaussian_presample_volsdf_backward_LOT(
                lotree,
                grad_out,
                color_cache + ray_id * 3,
                hit_num + ray_id,
                ray_spec[ray_blk_id],
                opt,
                lane_id,
                presample_t[ray_id].data(),
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                grads);
        }

        __launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_sdf_backward_kernel_LOT(
                PackedTreeSpecLOT lotree,
                PackedRaysHitLOTreeSDF rays_hls,
                int rays_hls_stride,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                PackedRaysSpec rays,
                RenderOptions opt,
                bool grad_out_is_rgb,
                PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
            const int ray_id = tid >> 5;
            const int ray_blk_id = threadIdx.x >> 5;
            const int lane_id = threadIdx.x & 0x1F;

            if (lane_id >= lotree.sh_data_dim)
                return;

            __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
            __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
            __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ SingleRayHitLOTreeSDF ray_hls[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
            __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
            ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                                     rays.dirs[ray_id].data());
            ray_hls[ray_blk_id].set(rays_hls.sdf_point + ray_id * rays_hls_stride * 3,
                                    rays_hls.col_point + ray_id * rays_hls_stride * 3,
                                    rays_hls.hitnode_sdf + ray_id * rays_hls_stride * 8,
                                    rays_hls.hitnode_col + ray_id * rays_hls_stride * 8,
                                    rays_hls.hitnum[ray_id]);
            const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                                   ray_spec[ray_blk_id].dir[1],
                                   ray_spec[ray_blk_id].dir[2]};
            if (lane_id < lotree.basis_dim)
            {
                grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
            }
            calc_sphfunc_LOT(lotree, vdir, sphfunc_val[ray_blk_id]);

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            __syncwarp((1U << lotree.sh_data_dim) - 1);
            trace_ray_sdf_backward_LOT(
                lotree,
                ray_hls[ray_blk_id],
                grad_out,
                color_cache + ray_id * 3,
                opt,
                lane_id,
                sphfunc_val[ray_blk_id],
                grad_sphfunc_val[ray_blk_id],
                temp_storage[ray_blk_id],
                grads);
        }

        __global__ void render_ray_eikonal_backward_kernel_LOT(
            PackedTreeSpecLOT lotree,
            PackedRaysHitLOTreeSDF rays_hls,
            int rays_hls_stride,
            int rays_size,
            const float *__restrict__ sgd_grad,
            float scale_grad,
            PackedGridOutputGradsSDFLOT grads)
        {
            CUDA_GET_THREAD_ID(tid, rays_size);

            trace_ray_eikonal_backward_LOT(
                lotree,
                rays_hls.sdf_point + tid * rays_hls_stride * 3,
                rays_hls.hitnode_sdf + tid * rays_hls_stride * 8,
                rays_hls.hitnum[tid],
                sgd_grad + tid * rays_hls_stride * 3,
                scale_grad,
                grads);
        }

        __launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
            __global__ void render_background_kernel(
                PackedSparseGridSpec grid,
                PackedRaysSpec rays,
                RenderOptions opt,
                const float *__restrict__ log_transmit,
                // Outputs
                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
        {
            CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
            if (log_transmit[ray_id] < -25.f)
                return;
            SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
            ray_find_bounds_bg(ray_spec, grid, opt, ray_id);
            render_background_forward(
                grid,
                ray_spec,
                opt,
                log_transmit[ray_id],
                out[ray_id].data());
        }

        __launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
            __global__ void render_background_image_kernel(
                PackedSparseGridSpec grid,
                PackedCameraSpec cam,
                RenderOptions opt,
                const float *__restrict__ log_transmit,
                // Outputs
                float *__restrict__ out)
        {
            CUDA_GET_THREAD_ID(ray_id, cam.height * cam.width);
            if (log_transmit[ray_id] < -25.f)
                return;
            const int ix = ray_id % cam.width;
            const int iy = ray_id / cam.width;
            SingleRaySpec ray_spec;
            cam2world_ray(ix, iy, cam, ray_spec.dir, ray_spec.origin);
            ray_find_bounds_bg(ray_spec, grid, opt, ray_id);
            render_background_forward(
                grid,
                ray_spec,
                opt,
                log_transmit[ray_id],
                out + ray_id * 3);
        }

        __launch_bounds__(TRACE_RAY_BG_CUDA_THREADS, MIN_BG_BLOCKS_PER_SM)
            __global__ void render_background_backward_kernel(
                PackedSparseGridSpec grid,
                const float *__restrict__ grad_output,
                const float *__restrict__ color_cache,
                PackedRaysSpec rays,
                RenderOptions opt,
                const float *__restrict__ log_transmit,
                const float *__restrict__ accum,
                bool grad_out_is_rgb,
                float sparsity_loss,
                // Outputs
                PackedGridOutputGrads grads)
        {
            CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
            if (log_transmit[ray_id] < -25.f)
                return;
            SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
            ray_find_bounds_bg(ray_spec, grid, opt, ray_id);

            float grad_out[3];
            if (grad_out_is_rgb)
            {
                const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
                    grad_out[i] = resid * norm_factor;
                }
            }
            else
            {
#pragma unroll 3
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] = grad_output[ray_id * 3 + i];
                }
            }

            render_background_backward(
                grid,
                grad_out,
                ray_spec,
                opt,
                log_transmit[ray_id],
                accum[ray_id],
                sparsity_loss,
                grads);
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_expected_term_kernel(
                PackedSparseGridSpec grid,
                PackedRaysSpec rays,
                RenderOptions opt,
                float *__restrict__ out)
        {
            // const PackedSparseGridSpec& __restrict__ grid,
            // SingleRaySpec& __restrict__ ray,
            // const RenderOptions& __restrict__ opt,
            // float* __restrict__ out) {
            CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
            SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
            ray_find_bounds(ray_spec, grid, opt, ray_id);
            trace_ray_expected_term(
                grid,
                ray_spec,
                opt,
                out + ray_id);
        }

        __launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void render_ray_sigma_thresh_kernel(
                PackedSparseGridSpec grid,
                PackedRaysSpec rays,
                RenderOptions opt,
                float sigma_thresh,
                float *__restrict__ out)
        {
            // const PackedSparseGridSpec& __restrict__ grid,
            // SingleRaySpec& __restrict__ ray,
            // const RenderOptions& __restrict__ opt,
            // float* __restrict__ out) {
            CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
            SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
            ray_find_bounds(ray_spec, grid, opt, ray_id);
            trace_ray_sigma_thresh(
                grid,
                ray_spec,
                opt,
                sigma_thresh,
                out + ray_id);
        }

    } // namespace device

    torch::Tensor _get_empty_1d(const torch::Tensor &origins)
    {
        auto options =
            torch::TensorOptions()
                .dtype(origins.dtype())
                .layout(torch::kStrided)
                .device(origins.device())
                .requires_grad(false);
        return torch::empty({origins.size(0)}, options);
    }

} // namespace

torch::Tensor volume_render_cuvol(SparseGridSpec &grid, RaysSpec &rays, RenderOptions &opt)
{
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    torch::Tensor results = torch::empty_like(rays.origins);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    torch::Tensor log_transmit;
    if (use_background)
    {
        log_transmit = _get_empty_1d(rays.origins);
    }

    {
        const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
        device::render_ray_kernel<<<blocks, cuda_n_threads>>>(
            grid, rays, opt,
            // Output
            results.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            use_background ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background)
    {
        // printf("RENDER BG\n");
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            log_transmit.data_ptr<float>(),
            results.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    CUDA_CHECK_ERRORS;
    return results;
}

torch::Tensor volume_render_cuvol_image(SparseGridSpec &grid, CameraSpec &cam, RenderOptions &opt)
{
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    cam.check();

    const auto Q = cam.height * cam.width;
    auto options =
        torch::TensorOptions()
            .dtype(grid.sh_data.dtype())
            .layout(torch::kStrided)
            .device(grid.sh_data.device())
            .requires_grad(false);

    torch::Tensor results = torch::empty({cam.height, cam.width, 3}, options);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    torch::Tensor log_transmit;
    if (use_background)
    {
        log_transmit = torch::empty({cam.height, cam.width}, options);
    }

    {
        const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
        device::render_ray_image_kernel<<<blocks, cuda_n_threads>>>(
            grid,
            cam,
            opt,
            // Output
            results.data_ptr<float>(),
            use_background ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background)
    {
        // printf("RENDER BG\n");
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_image_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
            grid,
            cam,
            opt,
            log_transmit.data_ptr<float>(),
            results.data_ptr<float>());
    }

    CUDA_CHECK_ERRORS;
    return results;
}

void volume_render_cuvol_backward(
    SparseGridSpec &grid,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor grad_out,
    torch::Tensor color_cache,
    GridOutputGrads &grads)
{

    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    torch::Tensor log_transmit, accum;
    if (use_background)
    {
        log_transmit = _get_empty_1d(rays.origins);
        accum = _get_empty_1d(rays.origins);
    }

    {
        const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);
        device::render_ray_backward_kernel<<<blocks,
                                             cuda_n_threads_render_backward>>>(
            grid,
            grad_out.data_ptr<float>(),
            color_cache.data_ptr<float>(),
            rays, opt,
            false,
            nullptr,
            0.f,
            0.f,
            // Output
            grads,
            use_background ? accum.data_ptr<float>() : nullptr,
            use_background ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background)
    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_backward_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
            grid,
            grad_out.data_ptr<float>(),
            color_cache.data_ptr<float>(),
            rays,
            opt,
            log_transmit.data_ptr<float>(),
            accum.data_ptr<float>(),
            false,
            0.f,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_cuvol_fused(
    SparseGridSpec &grid,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    float beta_loss,
    float sparsity_loss,
    torch::Tensor rgb_out,
    GridOutputGrads &grads)
{

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    bool need_log_transmit = use_background || beta_loss > 0.f;
    torch::Tensor log_transmit, accum;
    if (need_log_transmit)
    {
        log_transmit = _get_empty_1d(rays.origins);
    }
    if (use_background)
    {
        accum = _get_empty_1d(rays.origins);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            grid, rays, opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background)
    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            log_transmit.data_ptr<float>(),
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            grid,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            rays, opt,
            true,
            beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
            beta_loss / Q,
            sparsity_loss,
            // Output
            grads,
            use_background ? accum.data_ptr<float>() : nullptr,
            nullptr);
    }

    if (use_background)
    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BG_CUDA_THREADS);
        device::render_background_backward_kernel<<<blocks, TRACE_RAY_BG_CUDA_THREADS>>>(
            grid,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            rays,
            opt,
            log_transmit.data_ptr<float>(),
            accum.data_ptr<float>(),
            true,
            sparsity_loss,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_cuvol_fused_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    float beta_loss,
    float sparsity_loss,
    torch::Tensor rgb_out,
    GridOutputGradsLOT &grads)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool need_log_transmit = beta_loss > 0.f;
    torch::Tensor log_transmit;
    if (need_log_transmit)
    {
        log_transmit = _get_empty_1d(rays.origins);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel_LOT<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            lotree,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            rays, opt,
            true,
            beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
            beta_loss / Q,
            sparsity_loss,
            // Output
            grads,
            nullptr,
            nullptr);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_fused_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    torch::Tensor rgb_out,
    GridOutputGradsSDFLOT &grads)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_volsdf_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_volsdf_backward_kernel_LOT<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            lotree,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            rays, opt,
            true,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_convert_fused_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    torch::Tensor rgb_out,
    GridOutputGradsSDFLOT &grads)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_volsdf_convert_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_volsdf_convert_backward_kernel_LOT<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            lotree,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            rays, opt,
            true,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_downsample_fused_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    torch::Tensor rgb_out,
    torch::Tensor leaf_node_map,
    GridOutputGradsSDFLOT &grads)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_volsdf_downsample_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            // Output
            leaf_node_map.packed_accessor32<int32_t, 5, torch::RestrictPtrTraits>(),
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_volsdf_downsample_backward_kernel_LOT<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            lotree,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            leaf_node_map.data_ptr<int32_t>(),
            rays, opt,
            true,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_presample_volsdf_fused_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    torch::Tensor rgb_out,
    torch::Tensor hit_num,
    torch::Tensor presample_t,
    GridOutputGradsSDFLOT &grads)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(hit_num);
    lotree.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_presample_volsdf_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            presample_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_presample_volsdf_backward_kernel_LOT<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            lotree,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            hit_num.data_ptr<float>(),
            presample_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays, opt,
            true,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_gaussian_fused_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    torch::Tensor rgb_out,
    GridOutputGradsSDFLOT &grads)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_volsdf_gaussian_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_volsdf_gaussian_backward_kernel_LOT<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            lotree,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            rays, opt,
            true,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_presample_volsdf_gaussian_fused_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    torch::Tensor rgb_out,
    torch::Tensor hit_num,
    torch::Tensor presample_t,
    GridOutputGradsSDFLOT &grads)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(hit_num);
    lotree.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_presample_volsdf_gaussian_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            presample_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_presample_volsdf_gaussian_backward_kernel_LOT<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            lotree,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            hit_num.data_ptr<float>(),
            presample_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays, opt,
            true,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_record_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor record_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(record_out);
    lotree.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_volsdf_record_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays, opt,
        // Output
        record_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_record_w_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor record_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(record_out);
    lotree.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_volsdf_record_w_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays, opt,
        // Output
        record_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_presample_volsdf_record_w_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor presample_t,
    torch::Tensor record_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(record_out);
    lotree.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_presample_volsdf_record_w_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays, opt,
        presample_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // Output
        record_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_eikonal_fused_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    float scale,
    torch::Tensor uni_samples,
    torch::Tensor total_eikonal_loss,
    GridOutputGradsSDFLOT &grads)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(uni_samples);
    CHECK_INPUT(total_eikonal_loss);
    lotree.check();
    rays.check();
    grads.check();

    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    device::render_ray_volsdf_eikonal_kernel_LOT<<<blocks, cuda_n_threads>>>(
        lotree, rays, opt,
        scale,
        uni_samples.data_ptr<float>(),
        total_eikonal_loss.data_ptr<float>(),
        grads);

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_test_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_volsdf_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays, opt,
        // Output
        rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_convert_test_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_volsdf_convert_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays, opt,
        // Output
        rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_downsample_test_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor leaf_node_map,
    torch::Tensor rgb_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_volsdf_downsample_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays, opt,
        // Output
        leaf_node_map.packed_accessor32<int32_t, 5, torch::RestrictPtrTraits>(),
        rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_downsample_render_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_volsdf_downsample_render_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays, opt,
        // Output
        rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_presample_volsdf_test_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor presample_t,
    torch::Tensor rgb_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays.check();
    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_presample_volsdf_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays, opt,
        presample_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // Output
        rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_sdf_fused_LOT(
    TreeSpecLOT &lotree,
    RaysHitLOTreeSDF &rays_hls,
    int rays_hls_stride,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_gt,
    torch::Tensor rgb_out,
    GridOutputGradsSDFLOT &grads)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays_hls.check();
    rays.check();
    grads.check();

    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_sdf_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays,
            rays_hls, rays_hls_stride,
            opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_sdf_backward_kernel_LOT<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
            lotree,
            rays_hls,
            rays_hls_stride,
            rgb_gt.data_ptr<float>(),
            rgb_out.data_ptr<float>(),
            rays, opt,
            true,
            // Output
            grads);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_sdf_test_LOT(
    TreeSpecLOT &lotree,
    RaysHitLOTreeSDF &rays_hls,
    int rays_hls_stride,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_out)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    lotree.check();
    rays_hls.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
    device::render_ray_sdf_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        lotree, rays,
        rays_hls, rays_hls_stride,
        opt,
        // Output
        rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_eikonal_fused_LOT(
    TreeSpecLOT &lotree,
    RaysHitLOTreeSDF &rays_hls,
    int rays_hls_stride,
    int rays_size,
    torch::Tensor corner_sdf_grad,
    torch::Tensor sdf_grad_out,
    float scale_grad,
    GridOutputGradsSDFLOT &grads)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(corner_sdf_grad);

    lotree.check();
    rays_hls.check();
    grads.check();

    const auto Q = rays_hls.sdf_point.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    device::render_ray_eikonal_kernel_LOT<<<blocks, cuda_n_threads>>>(
        rays_hls,
        rays_hls_stride,
        rays_size,
        corner_sdf_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        // Output
        sdf_grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

    device::render_ray_eikonal_backward_kernel_LOT<<<blocks, cuda_n_threads>>>(
        lotree,
        rays_hls,
        rays_hls_stride,
        rays_size,
        sdf_grad_out.data_ptr<float>(),
        scale_grad,
        grads);
}

void volume_render_hitnum_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor hitnum_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(hitnum_out);
    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    device::render_ray_hitnum_kernel_LOT<<<blocks, cuda_n_threads>>>(
        lotree, rays, opt,
        // Output
        hitnum_out.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_hitnum_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor hitnum_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(hitnum_out);
    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    device::render_ray_volsdf_hitnum_kernel_LOT<<<blocks, cuda_n_threads>>>(
        lotree, rays, opt,
        // Output
        hitnum_out.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_colrefine_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor ray_mse,
    torch::Tensor node_mse_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(node_mse_out);
    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    device::render_ray_volsdf_colrefine_kernel_LOT<<<blocks, cuda_n_threads>>>(
        lotree, rays, opt,
        ray_mse.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        // Output
        node_mse_out.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_prehit_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor hitinfo_out)
{

    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(hitinfo_out);
    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    device::render_ray_volsdf_prehit_kernel_LOT<<<blocks, cuda_n_threads>>>(
        lotree, rays, opt,
        // Output
        hitinfo_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_hitpoint_sdf_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    float step,
    torch::Tensor sdf_point_out,
    torch::Tensor col_point_out,
    torch::Tensor hitnode_sdf_out,
    torch::Tensor hitnode_col_out,
    torch::Tensor hitnum_out)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(sdf_point_out);
    CHECK_INPUT(col_point_out);
    CHECK_INPUT(hitnode_sdf_out);
    CHECK_INPUT(hitnode_col_out);
    CHECK_INPUT(hitnum_out);
    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    const int cuda_n_threads = 512;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    device::render_ray_hitpoint_sdf_kernel_LOT<<<blocks, cuda_n_threads>>>(
        lotree, rays, step,
        // Output
        sdf_point_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        col_point_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        hitnode_sdf_out.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        hitnode_col_out.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
        hitnum_out.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

    CUDA_CHECK_ERRORS;
}

void volume_render_refine_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_out,
    torch::Tensor hitnode_out,
    torch::Tensor weight_out)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(hitnode_out);
    CHECK_INPUT(weight_out);

    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_refine_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            hitnode_out.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
            weight_out.packed_accessor32<float, 5, torch::RestrictPtrTraits>());
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_refine_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_out,
    torch::Tensor hitnode_out,
    torch::Tensor weight_out)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(hitnode_out);
    CHECK_INPUT(weight_out);

    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_volsdf_refine_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            hitnode_out.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
            weight_out.packed_accessor32<float, 5, torch::RestrictPtrTraits>());
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_volsdf_refine_sdf_LOT(
    TreeSpecLOT &lotree,
    RaysSpec &rays,
    RenderOptions &opt,
    torch::Tensor rgb_out,
    torch::Tensor hitnode_out,
    torch::Tensor sdf_out)
{
    DEVICE_GUARD(lotree.CornerSH);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(hitnode_out);
    CHECK_INPUT(sdf_out);

    lotree.check();
    rays.check();

    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_volsdf_refine_sdf_kernel_LOT<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            lotree, rays, opt,
            // Output
            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            hitnode_out.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
            sdf_out.packed_accessor32<float, 5, torch::RestrictPtrTraits>());
    }

    CUDA_CHECK_ERRORS;
}

torch::Tensor volume_render_expected_term(SparseGridSpec &grid,
                                          RaysSpec &rays, RenderOptions &opt)
{
    auto options =
        torch::TensorOptions()
            .dtype(rays.origins.dtype())
            .layout(torch::kStrided)
            .device(rays.origins.device())
            .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::render_ray_expected_term_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        grid,
        rays,
        opt,
        results.data_ptr<float>());
    return results;
}

torch::Tensor volume_render_sigma_thresh(SparseGridSpec &grid,
                                         RaysSpec &rays,
                                         RenderOptions &opt,
                                         float sigma_thresh)
{
    auto options =
        torch::TensorOptions()
            .dtype(rays.origins.dtype())
            .layout(torch::kStrided)
            .device(rays.origins.device())
            .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::render_ray_sigma_thresh_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
        grid,
        rays,
        opt,
        sigma_thresh,
        results.data_ptr<float>());
    return results;
}
