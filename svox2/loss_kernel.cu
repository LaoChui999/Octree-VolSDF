// Copyright 2021 Alex Yu
// Loss computation-related kernels

#include <torch/extension.h>
#include <cstdint>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include "cuda_util.cuh"
#include "render_util.cuh"
#include "data_spec_packed.cuh"

namespace
{

    const int WARP_SIZE = 32;
    const int TV_GRAD_CUDA_THREADS = 256;
    const int TV_GRAD_POINTS_PER_BLOCK = TV_GRAD_CUDA_THREADS / WARP_SIZE;
    const int MIN_BLOCKS_PER_SM = 4;

    typedef cub::WarpReduce<float> WarpReducef;

    namespace device
    {

        __device__ __inline__ void calculate_ray_scale(float ndc_coeffx,
                                                       float ndc_coeffy,
                                                       float z,
                                                       float maxx,
                                                       float maxy,
                                                       float maxz,
                                                       float *__restrict__ scale)
        {
            // if (ndc_coeffx > 0.f) {
            //     // FF NDC
            //     scale[0] = maxx * (1.f / 256.f);
            //     scale[1] = maxy * (1.f / 256.f);
            //     scale[2] = maxz * (1.f / 256.f);

            // The following shit does not work
            // // Normalized to [-1, 1] (with 0.5 padding)
            // // const float x_norm = (x + 0.5) / maxx * 2 - 1;
            // // const float y_norm = (y + 0.5) / maxy * 2 - 1;
            // const float z_norm = (z + 0.5) / maxz * 2 - 1;
            //
            // // NDC distances
            // const float disparity = (1 - z_norm) / 2.f; // in [0, 1]
            // scale[0] = (ndc_coeffx * disparity);
            // scale[1] = (ndc_coeffy * disparity);
            // scale[2] = -((z_norm - 1.f + 2.f / maxz) * disparity) / (maxz * 0.5f);
            // } else {
            scale[0] = maxx * (1.f / 256.f);
            scale[1] = maxy * (1.f / 256.f);
            scale[2] = maxz * (1.f / 256.f);
            // }
        }

#define CALCULATE_RAY_SCALE(out_name, maxx, maxy, maxz) \
    calculate_ray_scale(                                \
        ndc_coeffx, ndc_coeffy,                         \
        z,                                              \
        maxx,                                           \
        maxy,                                           \
        maxz,                                           \
        out_name)

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void tv_kernel(
                torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
                torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
                int start_dim, int end_dim,
                float scale,
                size_t Q,
                bool ignore_edge,
                float ndc_coeffx, float ndc_coeffy,
                // Output
                float *__restrict__ out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);

            typedef cub::BlockReduce<float, 1024> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;

            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int xyz = tid / (end_dim - start_dim);
            const int z = xyz % (links.size(2) - 1);
            const int xy = xyz / (links.size(2) - 1);
            const int y = xy % (links.size(1) - 1);
            const int x = xy / (links.size(1) - 1);

            if (ignore_edge && links[x][y][z] == 0)
                return;
            float scaling[3];
            CALCULATE_RAY_SCALE(scaling, links.size(0), links.size(1), links.size(2));

            const float val000 = (links[x][y][z] >= 0 ? data[links[x][y][z]][idx] : 0.f);
            const float null_val = (ignore_edge ? val000 : 0.f);
            const float val100 = (links[x + 1][y][z] >= 0 ? data[links[x + 1][y][z]][idx] : null_val);
            const float val010 = (links[x][y + 1][z] >= 0 ? data[links[x][y + 1][z]][idx] : null_val);
            const float val001 = (links[x][y][z + 1] >= 0 ? data[links[x][y][z + 1]][idx] : null_val);
            const float dx = (val100 - val000) * scaling[0];
            const float dy = (val010 - val000) * scaling[1];
            const float dz = (val001 - val000) * scaling[2];
            const float tresult = sqrtf(1e-5f + dx * dx + dy * dy + dz * dz);

            const float bresult = BlockReduce(temp_storage).Sum(tresult);
            if (threadIdx.x == 0)
            {
                atomicAdd(out, bresult * scale);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void tv_grad_kernel(
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
                int start_dim, int end_dim,
                float scale,
                size_t Q,
                bool ignore_edge,
                float ndc_coeffx, float ndc_coeffy,
                // Output
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            float dummy;
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int xyz = tid / (end_dim - start_dim);
            const int z = xyz % (links.size(2) - 1);
            const int xy = xyz / (links.size(2) - 1);
            const int y = xy % (links.size(1) - 1);
            const int x = xy / (links.size(1) - 1);

            if (ignore_edge && links[x][y][z] == 0)
                return;

            float scaling[3];
            CALCULATE_RAY_SCALE(scaling, links.size(0), links.size(1), links.size(2));

            const float *dptr = data.data();
            const size_t ddim = data.size(1);
            float v000 = 0.f, v100 = 0.f, v010 = 0.f, v001 = 0.f;
            float *gptr000 = &dummy,
                  *gptr100 = &dummy,
                  *gptr010 = &dummy,
                  *gptr001 = &dummy;

            if (links[x][y][z] >= 0)
            {
                const size_t lnk = links[x][y][z] * ddim + idx;
                v000 = dptr[lnk];
                gptr000 = grad_data + lnk;
            }
            if (links[x + 1][y][z] >= 0)
            {
                const size_t lnk = links[x + 1][y][z] * ddim + idx;
                v100 = dptr[lnk];
                gptr100 = grad_data + lnk;
            }
            else if (ignore_edge)
                v100 = v000;
            if (links[x][y + 1][z] >= 0)
            {
                const size_t lnk = links[x][y + 1][z] * ddim + idx;
                v010 = dptr[lnk];
                gptr010 = grad_data + lnk;
            }
            else if (ignore_edge)
                v010 = v000;
            if (links[x][y][z + 1] >= 0)
            {
                const size_t lnk = links[x][y][z + 1] * ddim + idx;
                v001 = dptr[lnk];
                gptr001 = grad_data + lnk;
            }
            else if (ignore_edge)
                v001 = v000;

            float dx = (v100 - v000);
            float dy = (v010 - v000);
            float dz = (v001 - v000);
            const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);
            dx *= scaling[0];
            dy *= scaling[1];
            dz *= scaling[2];
            if (dx != 0.f)
                atomicAdd(gptr100, dx * idelta);
            if (dy != 0.f)
                atomicAdd(gptr010, dy * idelta);
            if (dz != 0.f)
                atomicAdd(gptr001, dz * idelta);
            atomicAdd(gptr000, -(dx + dy + dz) * idelta);
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void tv_grad_sparse_kernel(
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> links,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
                const int32_t *__restrict__ rand_cells,
                int start_dim, int end_dim,
                float scale,
                size_t Q,
                bool ignore_edge,
                bool ignore_last_z,
                float ndc_coeffx, float ndc_coeffy,
                // Output
                bool *__restrict__ mask_out,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int xyz = rand_cells[tid / (end_dim - start_dim)];
            const int z = xyz % links.size(2);
            const int xy = xyz / links.size(2);
            const int y = xy % links.size(1);
            const int x = xy / links.size(1);

            const int32_t *__restrict__ links_ptr = &links[x][y][z];

            if (ignore_edge && *links_ptr == 0)
                return;

            float scaling[3];
            CALCULATE_RAY_SCALE(scaling, links.size(0), links.size(1), links.size(2));

            const int offx = links.stride(0), offy = links.stride(1);

            const auto lnk000 = links_ptr[0];
            const auto lnk001 = ((z + 1 < links.size(2)) &&
                                 (!ignore_last_z || z != links.size(2) - 2))
                                    ? links_ptr[1]
                                    : 0;
            const auto lnk010 = y + 1 < links.size(1) ? links_ptr[offy] : 0;
            const auto lnk100 = x + 1 < links.size(0) ? links_ptr[offx] : 0;
            if (ignore_last_z && z == links.size(2) - 2)
                return;

            const float v000 = lnk000 >= 0 ? data[lnk000][idx] : 0.f;
            const float null_val = (ignore_edge ? v000 : 0.f);
            const float v001 = lnk001 >= 0 ? data[lnk001][idx] : null_val,
                        v010 = lnk010 >= 0 ? data[lnk010][idx] : null_val,
                        v100 = lnk100 >= 0 ? data[lnk100][idx] : null_val;

            float dx = (v100 - v000);
            float dy = (v010 - v000);
            float dz = (v001 - v000);
            const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);

            dx *= scaling[0];
            dy *= scaling[1];
            dz *= scaling[2];

#define MAYBE_ADD_SET(lnk, val)                                       \
    if (lnk >= 0 && val != 0.f)                                       \
    {                                                                 \
        atomicAdd(&grad_data[lnk * data.size(1) + idx], val *idelta); \
        if (mask_out != nullptr)                                      \
        {                                                             \
            mask_out[lnk] = true;                                     \
        }                                                             \
    }

            const float sm = -(dx + dy + dz);
            MAYBE_ADD_SET(lnk000, sm);
            MAYBE_ADD_SET(lnk001, dz);
            MAYBE_ADD_SET(lnk010, dy);
            MAYBE_ADD_SET(lnk100, dx);

#undef MAYBE_ADD_SET
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void tv_grad_sparse_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_neighbors,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
                const int32_t *__restrict__ rand_cells,
                int start_dim, int end_dim,
                float scale,
                size_t Q,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_rand = rand_cells[tid / (end_dim - start_dim)];

            float scaling[3];
            scaling[0] = powf(2.f, node_neighbors[c_rand][3] + 1) * (1.f / 64.f);
            scaling[1] = powf(2.f, node_neighbors[c_rand][4] + 1) * (1.f / 64.f);
            scaling[2] = powf(2.f, node_neighbors[c_rand][5] + 1) * (1.f / 64.f);

            const float v000 = data[c_rand][idx];
            const float v100 = (node_neighbors[c_rand][0] != -1) ? data[node_neighbors[c_rand][0]][idx] : 0.f;
            const float v010 = (node_neighbors[c_rand][1] != -1) ? data[node_neighbors[c_rand][1]][idx] : 0.f;
            const float v001 = (node_neighbors[c_rand][2] != -1) ? data[node_neighbors[c_rand][2]][idx] : 0.f;

            float dx = (v100 - v000);
            float dy = (v010 - v000);
            float dz = (v001 - v000);
            const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);

            dx *= scaling[0];
            dy *= scaling[1];
            dz *= scaling[2];

            const float sm = -(dx + dy + dz);
            atomicAdd(&grad_data[c_rand * data.size(1) + idx], sm * idelta);
            atomicAdd(&grad_data[node_neighbors[c_rand][0] * data.size(1) + idx], dx * idelta);
            atomicAdd(&grad_data[node_neighbors[c_rand][1] * data.size(1) + idx], dy * idelta);
            atomicAdd(&grad_data[node_neighbors[c_rand][2] * data.size(1) + idx], dz * idelta);
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void tv_grad_sparse_thirdord_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
                const int32_t *__restrict__ rand_cells,
                const int32_t *__restrict__ geo_corner_map,
                int start_dim, int end_dim,
                float scale,
                size_t Q,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_rand = rand_cells[tid / (end_dim - start_dim)];

            const int g_index = geo_corner_map[c_rand];

            const float l100 = node_all_neighlen[g_index][1];
            const float l010 = node_all_neighlen[g_index][3];
            const float l001 = node_all_neighlen[g_index][5];

            if (l100 != 1.0 && l010 != 1.0 && l001 != 1.0)
            {
                const float v000 = data[c_rand][idx];

                float v_value[3];
                for (int i = 0; i < 3; ++i)
                {
                    int i_index = i * 2 + 1;
                    if (node_all_neighbors[g_index][i_index] != -1)
                    {
                        v_value[i] = data[node_all_neighbors[g_index][i_index]][idx];
                    }
                    else
                    {
                        if (node_gho_neighbors[g_index][i][3] == -1)
                        {
                            float v0 = data[node_gho_neighbors[g_index][i][0]][idx];
                            float v1 = data[node_gho_neighbors[g_index][i][1]][idx];

                            float c0 = node_gho_coeff[g_index][i][0];
                            float c1 = node_gho_coeff[g_index][i][1];

                            v_value[i] = c0 * v1 + c1 * v0;
                        }
                        else
                        {
                            float v0 = data[node_gho_neighbors[g_index][i][0]][idx];
                            float v1 = data[node_gho_neighbors[g_index][i][1]][idx];
                            float v2 = data[node_gho_neighbors[g_index][i][2]][idx];
                            float v3 = data[node_gho_neighbors[g_index][i][3]][idx];

                            float c0 = node_gho_coeff[g_index][i][0];
                            float c1 = node_gho_coeff[g_index][i][1];
                            float c2 = node_gho_coeff[g_index][i][2];
                            float c3 = node_gho_coeff[g_index][i][3];

                            v_value[i] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3;
                        }
                    }
                }

                float scaling[3];
                scaling[0] = (1.f / 64.f) / l100;
                scaling[1] = (1.f / 64.f) / l010;
                scaling[2] = (1.f / 64.f) / l001;

                float dxyz[3];
                dxyz[0] = (v_value[0] - v000);
                dxyz[1] = (v_value[1] - v000);
                dxyz[2] = (v_value[2] - v000);
                const float idelta = scale * rsqrtf(1e-9f + dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2]);

                dxyz[0] *= scaling[0];
                dxyz[1] *= scaling[1];
                dxyz[2] *= scaling[2];

                const float sm = -(dxyz[0] + dxyz[1] + dxyz[2]);
                atomicAdd(&grad_data[c_rand * data.size(1) + idx], sm * idelta);

                for (int i = 0; i < 3; ++i)
                {
                    int i_index = i * 2 + 1;
                    if (node_all_neighbors[g_index][i_index] != -1)
                    {
                        atomicAdd(&grad_data[node_all_neighbors[g_index][i_index] * data.size(1) + idx], dxyz[i] * idelta);
                    }
                    else
                    {
                        if (node_gho_neighbors[g_index][i][3] == -1)
                        {
                            float c0 = node_gho_coeff[g_index][i][0];
                            float c1 = node_gho_coeff[g_index][i][1];

                            atomicAdd(&grad_data[node_gho_neighbors[g_index][i][0] * data.size(1) + idx], dxyz[i] * idelta * c1);
                            atomicAdd(&grad_data[node_gho_neighbors[g_index][i][1] * data.size(1) + idx], dxyz[i] * idelta * c0);
                        }
                        else
                        {
                            float c0 = node_gho_coeff[g_index][i][0];
                            float c1 = node_gho_coeff[g_index][i][1];
                            float c2 = node_gho_coeff[g_index][i][2];
                            float c3 = node_gho_coeff[g_index][i][3];

                            atomicAdd(&grad_data[node_gho_neighbors[g_index][i][0] * data.size(1) + idx], dxyz[i] * idelta * c0);
                            atomicAdd(&grad_data[node_gho_neighbors[g_index][i][1] * data.size(1) + idx], dxyz[i] * idelta * c1);
                            atomicAdd(&grad_data[node_gho_neighbors[g_index][i][2] * data.size(1) + idx], dxyz[i] * idelta * c2);
                            atomicAdd(&grad_data[node_gho_neighbors[g_index][i][3] * data.size(1) + idx], dxyz[i] * idelta * c3);
                        }
                    }
                }
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void tv_grad_sparse_thirdord_mid_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
                const int32_t *__restrict__ rand_cells,
                const int32_t *__restrict__ geo_corner_map,
                int start_dim, int end_dim,
                float scale,
                size_t Q,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int32_t idx = tid % (end_dim - start_dim) + start_dim;

            int64_t index = int64_t(tid / int32_t(end_dim - start_dim));
            const int32_t c_index = rand_cells[index];

            int32_t g_index = geo_corner_map[c_index];

            const float l_100 = node_all_neighlen[g_index][0];
            const float l100 = node_all_neighlen[g_index][1];
            const float l_010 = node_all_neighlen[g_index][2];
            const float l010 = node_all_neighlen[g_index][3];
            const float l_001 = node_all_neighlen[g_index][4];
            const float l001 = node_all_neighlen[g_index][5];

            if (l_100 != 1.0 && l100 != 1.0 && l_010 != 1.0 && l010 != 1.0 && l_001 != 1.0 && l001 != 1.0)
            {
                const float v000 = data[c_index][idx];

                float v_value[6];
                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[g_index][i] != -1)
                    {
                        v_value[i] = data[node_all_neighbors[g_index][i]][idx];
                    }
                    else
                    {
                        int dir_index = floor((double)i / 2.0);
                        if (node_gho_neighbors[g_index][dir_index][3] == -1)
                        {
                            float v0 = data[node_gho_neighbors[g_index][dir_index][0]][idx];
                            float v1 = data[node_gho_neighbors[g_index][dir_index][1]][idx];

                            float c0 = node_gho_coeff[g_index][dir_index][0];
                            float c1 = node_gho_coeff[g_index][dir_index][1];

                            v_value[i] = c0 * v1 + c1 * v0;
                        }
                        else
                        {
                            float v0 = data[node_gho_neighbors[g_index][dir_index][0]][idx];
                            float v1 = data[node_gho_neighbors[g_index][dir_index][1]][idx];
                            float v2 = data[node_gho_neighbors[g_index][dir_index][2]][idx];
                            float v3 = data[node_gho_neighbors[g_index][dir_index][3]][idx];

                            float c0 = node_gho_coeff[g_index][dir_index][0];
                            float c1 = node_gho_coeff[g_index][dir_index][1];
                            float c2 = node_gho_coeff[g_index][dir_index][2];
                            float c3 = node_gho_coeff[g_index][dir_index][3];

                            v_value[i] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3;
                        }
                    }
                }

                float dxyz[3];
                dxyz[0] = ((v_value[1] - v000) / l100) * (l_100 / (l_100 + l100)) + ((v000 - v_value[0]) / l_100) * (l100 / (l_100 + l100));
                dxyz[1] = ((v_value[3] - v000) / l010) * (l_010 / (l_010 + l010)) + ((v000 - v_value[2]) / l_010) * (l010 / (l_010 + l010));
                dxyz[2] = ((v_value[5] - v000) / l001) * (l_001 / (l_001 + l001)) + ((v000 - v_value[4]) / l_001) * (l001 / (l_001 + l001));

                float xyz_grad_scale = scale * rsqrtf(1e-9f + dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2]);

                float dxyz_grad[6];
                dxyz_grad[0] = (-l100 / (l_100 * (l_100 + l100))) * dxyz[0];
                dxyz_grad[1] = (l_100 / (l100 * (l_100 + l100))) * dxyz[0];
                float dx_grad = -dxyz_grad[0] - dxyz_grad[1];

                dxyz_grad[2] = (-l010 / (l_010 * (l_010 + l010))) * dxyz[1];
                dxyz_grad[3] = (l_010 / (l010 * (l_010 + l010))) * dxyz[1];
                float dy_grad = -dxyz_grad[2] - dxyz_grad[3];

                dxyz_grad[4] = (-l001 / (l_001 * (l_001 + l001))) * dxyz[2];
                dxyz_grad[5] = (l_001 / (l001 * (l_001 + l001))) * dxyz[2];
                float dz_grad = -dxyz_grad[4] - dxyz_grad[5];

                const float sm = dx_grad + dy_grad + dz_grad;

                index = c_index * int64_t(data.size(1)) + int32_t(idx);
                atomicAdd(&grad_data[index], sm * xyz_grad_scale);

                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[g_index][i] != -1)
                    {
                        index = node_all_neighbors[g_index][i] * int64_t(data.size(1)) + int32_t(idx);
                        atomicAdd(&grad_data[index], dxyz_grad[i] * xyz_grad_scale);
                        continue;
                    }

                    int dir_index = floor((double)i / 2.0);

                    if (node_gho_neighbors[g_index][dir_index][3] == -1)
                    {
                        int g_i0 = node_gho_neighbors[g_index][dir_index][0];
                        int g_i1 = node_gho_neighbors[g_index][dir_index][1];

                        float c0 = node_gho_coeff[g_index][dir_index][0];
                        float c1 = node_gho_coeff[g_index][dir_index][1];

                        index = g_i0 * int64_t(data.size(1)) + int32_t(idx);
                        atomicAdd(&grad_data[index], dxyz_grad[i] * xyz_grad_scale * c1);

                        index = g_i1 * int64_t(data.size(1)) + int32_t(idx);
                        atomicAdd(&grad_data[index], dxyz_grad[i] * xyz_grad_scale * c0);
                    }
                    else
                    {
                        int g_i0 = node_gho_neighbors[g_index][dir_index][0];
                        int g_i1 = node_gho_neighbors[g_index][dir_index][1];
                        int g_i2 = node_gho_neighbors[g_index][dir_index][2];
                        int g_i3 = node_gho_neighbors[g_index][dir_index][3];

                        float c0 = node_gho_coeff[g_index][dir_index][0];
                        float c1 = node_gho_coeff[g_index][dir_index][1];
                        float c2 = node_gho_coeff[g_index][dir_index][2];
                        float c3 = node_gho_coeff[g_index][dir_index][3];

                        index = g_i0 * int64_t(data.size(1)) + int32_t(idx);
                        atomicAdd(&grad_data[index], dxyz_grad[i] * xyz_grad_scale * c0);
                        index = g_i1 * int64_t(data.size(1)) + int32_t(idx);
                        atomicAdd(&grad_data[index], dxyz_grad[i] * xyz_grad_scale * c1);
                        index = g_i2 * int64_t(data.size(1)) + int32_t(idx);
                        atomicAdd(&grad_data[index], dxyz_grad[i] * xyz_grad_scale * c2);
                        index = g_i3 * int64_t(data.size(1)) + int32_t(idx);
                        atomicAdd(&grad_data[index], dxyz_grad[i] * xyz_grad_scale * c3);
                    }
                }
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void sample_tri_interp_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_corners,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> sample_pos,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> low_pos,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
                const float *__restrict__ node_length,
                const int32_t *__restrict__ rand_cells,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ out_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int32_t idx = tid % (end_dim - start_dim) + start_dim;

            int64_t index = int64_t(tid / int32_t(end_dim - start_dim));
            const int32_t c_index = rand_cells[index];

            float sample_data_avg = 0.0;
            for (int sample_index = 0; sample_index < sample_pos.size(1); ++sample_index)
            {
                float pos[3];
                for (int i = 0; i < 3; ++i)
                {
                    pos[i] = (sample_pos[c_index][sample_index][i] - low_pos[c_index][i]) / node_length[c_index];
                }

                float pos_d[3];
                for (int i = 0; i < 3; ++i)
                {
                    pos_d[i] = 1.0 - pos[i];
                }

                float c00 = data[node_corners[c_index][0]][idx] * pos_d[0] + data[node_corners[c_index][2]][idx] * pos[0];
                float c01 = data[node_corners[c_index][4]][idx] * pos_d[0] + data[node_corners[c_index][6]][idx] * pos[0];
                float c10 = data[node_corners[c_index][1]][idx] * pos_d[0] + data[node_corners[c_index][3]][idx] * pos[0];
                float c11 = data[node_corners[c_index][5]][idx] * pos_d[0] + data[node_corners[c_index][7]][idx] * pos[0];
                float c0 = c00 * pos_d[1] + c10 * pos[1];
                float c1 = c01 * pos_d[1] + c11 * pos[1];
                float sample_data = c0 * pos_d[2] + c1 * pos[2];

                sample_data_avg += sample_data;
            }
            sample_data_avg /= ((float)sample_pos.size(1));

            index = c_index * int64_t(data.size(1)) + int32_t(idx);
            atomicAdd(&out_data[index], sample_data_avg);
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void sample_tri_interp_min_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_corners,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> sample_pos,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> low_pos,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> data,
                const float *__restrict__ node_length,
                const int32_t *__restrict__ rand_cells,
                float sdf_thresh,
                float sdf_offset,
                bool is_abs,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ out_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int32_t idx = tid % (end_dim - start_dim) + start_dim;

            int64_t index = int64_t(tid / int32_t(end_dim - start_dim));
            const int32_t c_index = rand_cells[index];

            float sample_data_avg = 0.0;
            int64_t o_index = c_index * int64_t(data.size(1)) + int32_t(idx);
            for (int sample_index = 0; sample_index < sample_pos.size(1); ++sample_index)
            {
                float pos[3];
                for (int i = 0; i < 3; ++i)
                {
                    pos[i] = (sample_pos[c_index][sample_index][i] - low_pos[c_index][i]) / node_length[c_index];
                }

                float pos_d[3];
                for (int i = 0; i < 3; ++i)
                {
                    pos_d[i] = 1.0 - pos[i];
                }

                float c00 = data[node_corners[c_index][0]][idx] * pos_d[0] + data[node_corners[c_index][2]][idx] * pos[0];
                float c01 = data[node_corners[c_index][4]][idx] * pos_d[0] + data[node_corners[c_index][6]][idx] * pos[0];
                float c10 = data[node_corners[c_index][1]][idx] * pos_d[0] + data[node_corners[c_index][3]][idx] * pos[0];
                float c11 = data[node_corners[c_index][5]][idx] * pos_d[0] + data[node_corners[c_index][7]][idx] * pos[0];
                float c0 = c00 * pos_d[1] + c10 * pos[1];
                float c1 = c01 * pos_d[1] + c11 * pos[1];
                float sample_data = c0 * pos_d[2] + c1 * pos[2];

                if((-sample_data) >= sdf_thresh)
                {
                    sample_data += sdf_offset;
                }

                if (is_abs)
                {
                    sample_data = fabsf(sample_data);
                }

                atomicMin(&out_data[o_index], sample_data);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void sdf_grad_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ sdf_grad)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float v000 = sdf_data[c_index][idx];

            const float v_100 = (node_all_neighbors[c_index][0] != -1) ? sdf_data[node_all_neighbors[c_index][0]][idx] : 0.f;
            const float v100 = (node_all_neighbors[c_index][1] != -1) ? sdf_data[node_all_neighbors[c_index][1]][idx] : 0.f;
            const float v_010 = (node_all_neighbors[c_index][2] != -1) ? sdf_data[node_all_neighbors[c_index][2]][idx] : 0.f;
            const float v010 = (node_all_neighbors[c_index][3] != -1) ? sdf_data[node_all_neighbors[c_index][3]][idx] : 0.f;
            const float v_001 = (node_all_neighbors[c_index][4] != -1) ? sdf_data[node_all_neighbors[c_index][4]][idx] : 0.f;
            const float v001 = (node_all_neighbors[c_index][5] != -1) ? sdf_data[node_all_neighbors[c_index][5]][idx] : 0.f;

            const float l_100 = (node_all_neighbors[c_index][0] != -1) ? node_all_neighlen[c_index][0] : 0.f;
            const float l100 = (node_all_neighbors[c_index][1] != -1) ? node_all_neighlen[c_index][1] : 0.f;
            const float l_010 = (node_all_neighbors[c_index][2] != -1) ? node_all_neighlen[c_index][2] : 0.f;
            const float l010 = (node_all_neighbors[c_index][3] != -1) ? node_all_neighlen[c_index][3] : 0.f;
            const float l_001 = (node_all_neighbors[c_index][4] != -1) ? node_all_neighlen[c_index][4] : 0.f;
            const float l001 = (node_all_neighbors[c_index][5] != -1) ? node_all_neighlen[c_index][5] : 0.f;

            if (l_100 != 0.0 && l100 != 0.0)
            {
                float total_len_x = l_100 + l100;
                sdf_grad[c_index * 3] = (l_100 / total_len_x) * ((v100 - v000) / l100) +
                                        (l100 / total_len_x) * ((v000 - v_100) / l_100);
            }
            else if (l_100 != 0.0)
            {
                sdf_grad[c_index * 3] = (v000 - v_100) / l_100;
            }
            else if (l100 != 0.0)
            {
                sdf_grad[c_index * 3] = (v100 - v000) / l100;
            }

            if (l_010 != 0.0 && l010 != 0.0)
            {
                float total_len_y = l_010 + l010;
                sdf_grad[c_index * 3 + 1] = (l_010 / total_len_y) * ((v010 - v000) / l010) +
                                            (l010 / total_len_y) * ((v000 - v_010) / l_010);
            }
            else if (l_010 != 0.0)
            {
                sdf_grad[c_index * 3 + 1] = (v000 - v_010) / l_010;
            }
            else if (l010 != 0.0)
            {
                sdf_grad[c_index * 3 + 1] = (v010 - v000) / l010;
            }

            if (l_001 != 0.0 && l001 != 0.0)
            {
                float total_len_z = l_001 + l001;
                sdf_grad[c_index * 3 + 2] = (l_001 / total_len_z) * ((v001 - v000) / l001) +
                                            (l001 / total_len_z) * ((v000 - v_001) / l_001);
            }
            else if (l_001 != 0.0)
            {
                sdf_grad[c_index * 3 + 2] = (v000 - v_001) / l_001;
            }
            else if (l001 != 0.0)
            {
                sdf_grad[c_index * 3 + 2] = (v001 - v000) / l001;
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void uni_sdf_grad_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ sdf_grad)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float v000 = sdf_data[c_index][idx];

            const float v_100 = (node_all_neighbors[c_index][0] != -1) ? sdf_data[node_all_neighbors[c_index][0]][idx] : 0.f;
            const float v100 = (node_all_neighbors[c_index][1] != -1) ? sdf_data[node_all_neighbors[c_index][1]][idx] : 0.f;
            const float v_010 = (node_all_neighbors[c_index][2] != -1) ? sdf_data[node_all_neighbors[c_index][2]][idx] : 0.f;
            const float v010 = (node_all_neighbors[c_index][3] != -1) ? sdf_data[node_all_neighbors[c_index][3]][idx] : 0.f;
            const float v_001 = (node_all_neighbors[c_index][4] != -1) ? sdf_data[node_all_neighbors[c_index][4]][idx] : 0.f;
            const float v001 = (node_all_neighbors[c_index][5] != -1) ? sdf_data[node_all_neighbors[c_index][5]][idx] : 0.f;

            const float l_100 = (node_all_neighbors[c_index][0] != -1) ? node_all_neighlen[c_index][0] : 0.f;
            const float l100 = (node_all_neighbors[c_index][1] != -1) ? node_all_neighlen[c_index][1] : 0.f;
            const float l_010 = (node_all_neighbors[c_index][2] != -1) ? node_all_neighlen[c_index][2] : 0.f;
            const float l010 = (node_all_neighbors[c_index][3] != -1) ? node_all_neighlen[c_index][3] : 0.f;
            const float l_001 = (node_all_neighbors[c_index][4] != -1) ? node_all_neighlen[c_index][4] : 0.f;
            const float l001 = (node_all_neighbors[c_index][5] != -1) ? node_all_neighlen[c_index][5] : 0.f;

            if (l_100 != 0.0 && l100 != 0.0)
            {
                float total_len_x = l_100 + l100;
                sdf_grad[c_index * 3] = 0.5 * ((v100 - v000) / l100) +
                                        0.5 * ((v000 - v_100) / l_100);
            }
            else if (l_100 != 0.0)
            {
                sdf_grad[c_index * 3] = (v000 - v_100) / l_100;
            }
            else if (l100 != 0.0)
            {
                sdf_grad[c_index * 3] = (v100 - v000) / l100;
            }

            if (l_010 != 0.0 && l010 != 0.0)
            {
                float total_len_y = l_010 + l010;
                sdf_grad[c_index * 3 + 1] = 0.5 * ((v010 - v000) / l010) +
                                            0.5 * ((v000 - v_010) / l_010);
            }
            else if (l_010 != 0.0)
            {
                sdf_grad[c_index * 3 + 1] = (v000 - v_010) / l_010;
            }
            else if (l010 != 0.0)
            {
                sdf_grad[c_index * 3 + 1] = (v010 - v000) / l010;
            }

            if (l_001 != 0.0 && l001 != 0.0)
            {
                float total_len_z = l_001 + l001;
                sdf_grad[c_index * 3 + 2] = 0.5 * ((v001 - v000) / l001) +
                                            0.5 * ((v000 - v_001) / l_001);
            }
            else if (l_001 != 0.0)
            {
                sdf_grad[c_index * 3 + 2] = (v000 - v_001) / l_001;
            }
            else if (l001 != 0.0)
            {
                sdf_grad[c_index * 3 + 2] = (v001 - v000) / l001;
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void uni_sdf_grad_backward_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_grad,
                const int32_t *__restrict__ corner_indices,
                float scale,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float g000_x = sdf_grad[c_index][0];
            const float g000_y = sdf_grad[c_index][1];
            const float g000_z = sdf_grad[c_index][2];

            float square_grad = sqrt(g000_x * g000_x + g000_y * g000_y + g000_z * g000_z + 1e-8);
            float eikonal_loss_sqrt = square_grad - 1.0;
            float square_grad_inv = 1.0 / square_grad;

            float grad_scale = eikonal_loss_sqrt * square_grad_inv * 2.0 * scale;

            const float l_100 = (node_all_neighbors[c_index][0] != -1) ? node_all_neighlen[c_index][0] : 0.f;
            const float l100 = (node_all_neighbors[c_index][1] != -1) ? node_all_neighlen[c_index][1] : 0.f;
            const float l_010 = (node_all_neighbors[c_index][2] != -1) ? node_all_neighlen[c_index][2] : 0.f;
            const float l010 = (node_all_neighbors[c_index][3] != -1) ? node_all_neighlen[c_index][3] : 0.f;
            const float l_001 = (node_all_neighbors[c_index][4] != -1) ? node_all_neighlen[c_index][4] : 0.f;
            const float l001 = (node_all_neighbors[c_index][5] != -1) ? node_all_neighlen[c_index][5] : 0.f;

            if (l_100 != 0.0 && l100 != 0.0)
            {
                float total_len_x = l_100 + l100;
                atomicAdd(&grad_data[node_all_neighbors[c_index][0]], grad_scale * g000_x * (-1.0 / total_len_x));
                atomicAdd(&grad_data[node_all_neighbors[c_index][1]], grad_scale * g000_x * (1.0 / total_len_x));
            }
            else if (l_100 != 0.0)
            {
                atomicAdd(&grad_data[node_all_neighbors[c_index][0]], grad_scale * g000_x * (-1.0 / l_100));
                atomicAdd(&grad_data[c_index], grad_scale * g000_x * (1.0 / l_100));
            }
            else if (l100 != 0.0)
            {
                atomicAdd(&grad_data[node_all_neighbors[c_index][1]], grad_scale * g000_x * (1.0 / l100));
                atomicAdd(&grad_data[c_index], grad_scale * g000_x * (-1.0 / l100));
            }

            if (l_010 != 0.0 && l010 != 0.0)
            {
                float total_len_y = l_010 + l010;
                atomicAdd(&grad_data[node_all_neighbors[c_index][2]], grad_scale * g000_y * (-1.0 / total_len_y));
                atomicAdd(&grad_data[node_all_neighbors[c_index][3]], grad_scale * g000_y * (1.0 / total_len_y));
            }
            else if (l_010 != 0.0)
            {
                atomicAdd(&grad_data[node_all_neighbors[c_index][2]], grad_scale * g000_y * (-1.0 / l_010));
                atomicAdd(&grad_data[c_index], grad_scale * g000_y * (1.0 / l_010));
            }
            else if (l010 != 0.0)
            {
                atomicAdd(&grad_data[node_all_neighbors[c_index][3]], grad_scale * g000_y * (1.0 / l010));
                atomicAdd(&grad_data[c_index], grad_scale * g000_y * (-1.0 / l010));
            }

            if (l_001 != 0.0 && l001 != 0.0)
            {
                float total_len_z = l_001 + l001;
                atomicAdd(&grad_data[node_all_neighbors[c_index][4]], grad_scale * g000_z * (-1.0 / total_len_z));
                atomicAdd(&grad_data[node_all_neighbors[c_index][5]], grad_scale * g000_z * (1.0 / total_len_z));
            }
            else if (l_001 != 0.0)
            {
                atomicAdd(&grad_data[node_all_neighbors[c_index][4]], grad_scale * g000_z * (-1.0 / l_001));
                atomicAdd(&grad_data[c_index], grad_scale * g000_z * (1.0 / l_001));
            }
            else if (l001 != 0.0)
            {
                atomicAdd(&grad_data[node_all_neighbors[c_index][5]], grad_scale * g000_z * (1.0 / l001));
                atomicAdd(&grad_data[c_index], grad_scale * g000_z * (-1.0 / l001));
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void uni_viscosity_loss_fused_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                float scale,
                float epsilon,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ total_eikonal_loss,
                float *__restrict__ total_viscosity_loss,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float l_100 = (node_all_neighbors[c_index][0] != -1) ? node_all_neighlen[c_index][0] : 0.f;
            const float l100 = (node_all_neighbors[c_index][1] != -1) ? node_all_neighlen[c_index][1] : 0.f;
            const float l_010 = (node_all_neighbors[c_index][2] != -1) ? node_all_neighlen[c_index][2] : 0.f;
            const float l010 = (node_all_neighbors[c_index][3] != -1) ? node_all_neighlen[c_index][3] : 0.f;
            const float l_001 = (node_all_neighbors[c_index][4] != -1) ? node_all_neighlen[c_index][4] : 0.f;
            const float l001 = (node_all_neighbors[c_index][5] != -1) ? node_all_neighlen[c_index][5] : 0.f;

            if (l_100 != 0.0 && l100 != 0.0 && l_010 != 0.0 && l010 != 0.0 && l_001 != 0.0 && l001 != 0.0)
            {
                const float v000 = sdf_data[c_index][idx];

                const float v_100 = (node_all_neighbors[c_index][0] != -1) ? sdf_data[node_all_neighbors[c_index][0]][idx] : 0.f;
                const float v100 = (node_all_neighbors[c_index][1] != -1) ? sdf_data[node_all_neighbors[c_index][1]][idx] : 0.f;
                const float v_010 = (node_all_neighbors[c_index][2] != -1) ? sdf_data[node_all_neighbors[c_index][2]][idx] : 0.f;
                const float v010 = (node_all_neighbors[c_index][3] != -1) ? sdf_data[node_all_neighbors[c_index][3]][idx] : 0.f;
                const float v_001 = (node_all_neighbors[c_index][4] != -1) ? sdf_data[node_all_neighbors[c_index][4]][idx] : 0.f;
                const float v001 = (node_all_neighbors[c_index][5] != -1) ? sdf_data[node_all_neighbors[c_index][5]][idx] : 0.f;

                float total_len_x = l_100 + l100;
                float Dx = (v100 - v_100) / total_len_x;
                float D2x = (v100 - 2.0 * v000 + v_100) / (l100 * l100);

                float total_len_y = l_010 + l010;
                float Dy = (v010 - v_010) / total_len_y;
                float D2y = (v010 - 2.0 * v000 + v_010) / (l010 * l010);

                float total_len_z = l_001 + l001;
                float Dz = (v001 - v_001) / total_len_z;
                float D2z = (v001 - 2.0 * v000 + v_001) / (l001 * l001);

                float sdf_grad_len = sqrt(Dx * Dx + Dy * Dy + Dz * Dz + 1e-8);
                float eikonal_loss = sdf_grad_len - 1.0;
                float sdf_sign = (v000 >= 0.0) ? 1.0 : -1.0;
                float viscosity_loss_sqrt = sdf_sign * eikonal_loss - epsilon * (D2x + D2y + D2z);
                float viscosity_loss = viscosity_loss_sqrt * viscosity_loss_sqrt;

                float sdf_grad_len_inv = 1.0 / sdf_grad_len;
                float viscosity_loss_0 = sdf_sign * sdf_grad_len_inv;

                float viscosity_loss_x0 = viscosity_loss_0 * Dx * (-1.0 / total_len_x) - epsilon * (1.0 / (l100 * l100));
                float viscosity_loss_x = epsilon * (2.0 / (l100 * l100));
                float viscosity_loss_x1 = viscosity_loss_0 * Dx * (1.0 / total_len_x) - epsilon * (1.0 / (l100 * l100));

                float viscosity_loss_y0 = viscosity_loss_0 * Dy * (-1.0 / total_len_y) - epsilon * (1.0 / (l010 * l010));
                float viscosity_loss_y = epsilon * (2.0 / (l010 * l010));
                float viscosity_loss_y1 = viscosity_loss_0 * Dy * (1.0 / total_len_y) - epsilon * (1.0 / (l010 * l010));

                float viscosity_loss_z0 = viscosity_loss_0 * Dz * (-1.0 / total_len_z) - epsilon * (1.0 / (l001 * l001));
                float viscosity_loss_z = epsilon * (2.0 / (l001 * l001));
                float viscosity_loss_z1 = viscosity_loss_0 * Dz * (1.0 / total_len_z) - epsilon * (1.0 / (l001 * l001));

                float viscosity_loss_scale = scale * 2.0 * viscosity_loss_sqrt;

                float viscosity_loss_grad = (viscosity_loss_x + viscosity_loss_y + viscosity_loss_z) * viscosity_loss_scale;

                atomicAdd(&grad_data[c_index], viscosity_loss_grad);
                atomicAdd(&grad_data[node_all_neighbors[c_index][0]], viscosity_loss_x0 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][1]], viscosity_loss_x1 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][2]], viscosity_loss_y0 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][3]], viscosity_loss_y1 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][4]], viscosity_loss_z0 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][5]], viscosity_loss_z1 * viscosity_loss_scale);

                atomicAdd(&total_eikonal_loss[0], eikonal_loss * eikonal_loss);
                atomicAdd(&total_viscosity_loss[0], viscosity_loss);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void uni_laplacian_loss_fused_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                float scale,
                float epsilon,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ total_laplacian_loss,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float l_100 = (node_all_neighbors[c_index][0] != -1) ? node_all_neighlen[c_index][0] : 0.f;
            const float l100 = (node_all_neighbors[c_index][1] != -1) ? node_all_neighlen[c_index][1] : 0.f;
            const float l_010 = (node_all_neighbors[c_index][2] != -1) ? node_all_neighlen[c_index][2] : 0.f;
            const float l010 = (node_all_neighbors[c_index][3] != -1) ? node_all_neighlen[c_index][3] : 0.f;
            const float l_001 = (node_all_neighbors[c_index][4] != -1) ? node_all_neighlen[c_index][4] : 0.f;
            const float l001 = (node_all_neighbors[c_index][5] != -1) ? node_all_neighlen[c_index][5] : 0.f;

            if (l_100 != 0.0 && l100 != 0.0 && l_010 != 0.0 && l010 != 0.0 && l_001 != 0.0 && l001 != 0.0)
            {
                const float v000 = sdf_data[c_index][idx];

                const float v_100 = (node_all_neighbors[c_index][0] != -1) ? sdf_data[node_all_neighbors[c_index][0]][idx] : 0.f;
                const float v100 = (node_all_neighbors[c_index][1] != -1) ? sdf_data[node_all_neighbors[c_index][1]][idx] : 0.f;
                const float v_010 = (node_all_neighbors[c_index][2] != -1) ? sdf_data[node_all_neighbors[c_index][2]][idx] : 0.f;
                const float v010 = (node_all_neighbors[c_index][3] != -1) ? sdf_data[node_all_neighbors[c_index][3]][idx] : 0.f;
                const float v_001 = (node_all_neighbors[c_index][4] != -1) ? sdf_data[node_all_neighbors[c_index][4]][idx] : 0.f;
                const float v001 = (node_all_neighbors[c_index][5] != -1) ? sdf_data[node_all_neighbors[c_index][5]][idx] : 0.f;

                float D2x = (v100 - 2.0 * v000 + v_100) / (l100 * l100);
                float D2y = (v010 - 2.0 * v000 + v_010) / (l010 * l010);
                float D2z = (v001 - 2.0 * v000 + v_001) / (l001 * l001);

                float laplacian_loss_sqrt = D2x + D2y + D2z;
                // float laplacian_loss = laplacian_loss_sqrt * laplacian_loss_sqrt;
                float sign = 1.0;
                if (laplacian_loss_sqrt < 0)
                {
                    sign = -1.0;
                    laplacian_loss_sqrt = -laplacian_loss_sqrt;
                }

                /*
                float laplacian_loss_x0 = 1.0 / (l100 * l100) * laplacian_loss_sqrt * 2.0;
                float laplacian_loss_x = -2.0 / (l100 * l100) * laplacian_loss_sqrt * 2.0;
                float laplacian_loss_x1 = 1.0 / (l100 * l100) * laplacian_loss_sqrt * 2.0;

                float laplacian_loss_y0 = 1.0 / (l010 * l010) * laplacian_loss_sqrt * 2.0;
                float laplacian_loss_y = -2.0 / (l010 * l010) * laplacian_loss_sqrt * 2.0;
                float laplacian_loss_y1 = 1.0 / (l010 * l010) * laplacian_loss_sqrt * 2.0;

                float laplacian_loss_z0 = 1.0 / (l001 * l001) * laplacian_loss_sqrt * 2.0;
                float laplacian_loss_z = -2.0 / (l001 * l001) * laplacian_loss_sqrt * 2.0;
                float laplacian_loss_z1 = 1.0 / (l001 * l001) * laplacian_loss_sqrt * 2.0;
                */

                float laplacian_loss_x0 = 1.0 / (l100 * l100) * sign;
                float laplacian_loss_x = -2.0 / (l100 * l100) * sign;
                float laplacian_loss_x1 = 1.0 / (l100 * l100) * sign;

                float laplacian_loss_y0 = 1.0 / (l010 * l010) * sign;
                float laplacian_loss_y = -2.0 / (l010 * l010) * sign;
                float laplacian_loss_y1 = 1.0 / (l010 * l010) * sign;

                float laplacian_loss_z0 = 1.0 / (l001 * l001) * sign;
                float laplacian_loss_z = -2.0 / (l001 * l001) * sign;
                float laplacian_loss_z1 = 1.0 / (l001 * l001) * sign;

                // float laplacian_loss_scale = scale * 2.0 * laplacian_loss_sqrt;

                float laplacian_loss_grad = laplacian_loss_x + laplacian_loss_y + laplacian_loss_z;

                atomicAdd(&grad_data[c_index], laplacian_loss_grad * scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][0]], laplacian_loss_x0 * scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][1]], laplacian_loss_x1 * scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][2]], laplacian_loss_y0 * scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][3]], laplacian_loss_y1 * scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][4]], laplacian_loss_z0 * scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][5]], laplacian_loss_z1 * scale);

                // atomicAdd(&total_laplacian_loss[0], laplacian_loss_sqrt * laplacian_loss_sqrt);
                atomicAdd(&total_laplacian_loss[0], laplacian_loss_sqrt);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void laplacian_loss_fused_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                float scale,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ total_laplacian_loss,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float l_100 = node_all_neighlen[c_index][0];
            const float l100 = node_all_neighlen[c_index][1];
            const float l_010 = node_all_neighlen[c_index][2];
            const float l010 = node_all_neighlen[c_index][3];
            const float l_001 = node_all_neighlen[c_index][4];
            const float l001 = node_all_neighlen[c_index][5];

            if (l_100 != 1.0 && l100 != 1.0 && l_010 != 1.0 && l010 != 1.0 && l_001 != 1.0 && l001 != 1.0)
            {
                const float v000 = sdf_data[c_index][idx];

                float v_value[6];
                int ghost_a_cor_index = -1;
                int ghost_b_cor_index = -1;
                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[c_index][i] != -1)
                    {
                        v_value[i] = sdf_data[node_all_neighbors[c_index][i]][idx];
                    }
                    else
                    {
                        int dir_index = floor((double)i / 2.0);
                        if (node_gho_neighbors[c_index][dir_index][3] == -1)
                        {
                            float v0 = sdf_data[node_gho_neighbors[c_index][dir_index][0]][idx];
                            float v1 = sdf_data[node_gho_neighbors[c_index][dir_index][1]][idx];

                            int pos_index = node_gho_neighbors[c_index][dir_index][2];
                            float v2 = sdf_data[node_all_neighbors[c_index][pos_index * 2]][idx];
                            float v3 = sdf_data[node_all_neighbors[c_index][pos_index * 2 + 1]][idx];

                            float c0 = node_gho_coeff[c_index][dir_index][0];
                            float c1 = node_gho_coeff[c_index][dir_index][1];
                            float c2 = node_gho_coeff[c_index][dir_index][2];
                            float c3 = node_gho_coeff[c_index][dir_index][3];
                            float c4 = node_gho_coeff[c_index][dir_index][4];

                            v_value[i] = c0 * v1 + c1 * v0 - c2 * v2 - c3 * v3 + c4 * v000;

                            ghost_a_cor_index = i;
                        }
                        else
                        {
                            ghost_b_cor_index = i;
                        }
                    }
                }

                if (ghost_b_cor_index != -1)
                {
                    int dir_index = floor((double)ghost_b_cor_index / 2.0);

                    int pos_index[2];
                    if (dir_index == 0)
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 2;
                    }
                    else if (dir_index == 1)
                    {
                        pos_index[0] = 0;
                        pos_index[1] = 2;
                    }
                    else
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 0;
                    }

                    float v0 = sdf_data[node_gho_neighbors[c_index][dir_index][0]][idx];
                    float v1 = sdf_data[node_gho_neighbors[c_index][dir_index][1]][idx];
                    float v2 = sdf_data[node_gho_neighbors[c_index][dir_index][2]][idx];
                    float v3 = sdf_data[node_gho_neighbors[c_index][dir_index][3]][idx];
                    float v4 = v_value[pos_index[0] * 2];
                    float v5 = v_value[pos_index[0] * 2 + 1];
                    float v6 = v_value[pos_index[1] * 2];
                    float v7 = v_value[pos_index[1] * 2 + 1];

                    float c0 = node_gho_coeff[c_index][dir_index][0];
                    float c1 = node_gho_coeff[c_index][dir_index][1];
                    float c2 = node_gho_coeff[c_index][dir_index][2];
                    float c3 = node_gho_coeff[c_index][dir_index][3];
                    float c4 = node_gho_coeff[c_index][dir_index][4];
                    float c5 = node_gho_coeff[c_index][dir_index][5];
                    float c6 = node_gho_coeff[c_index][dir_index][6];
                    float c7 = node_gho_coeff[c_index][dir_index][7];
                    float c8 = node_gho_coeff[c_index][dir_index][8];

                    v_value[ghost_b_cor_index] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3 - c4 * v4 - c5 * v5 - c6 * v6 - c7 * v7 + c8 * v000;
                }

                float D2x = ((v_value[1] - v000) / l100) * (2.0 / (l100 + l_100)) - ((v000 - v_value[0]) / l_100) * (2.0 / (l100 + l_100));
                float D2y = ((v_value[3] - v000) / l010) * (2.0 / (l010 + l_010)) - ((v000 - v_value[2]) / l_010) * (2.0 / (l010 + l_010));
                float D2z = ((v_value[5] - v000) / l001) * (2.0 / (l001 + l_001)) - ((v000 - v_value[4]) / l_001) * (2.0 / (l001 + l_001));

                float laplacian_loss_sqrt = D2x + D2y + D2z;

                float sign = 1.0;
                if (laplacian_loss_sqrt < 0)
                {
                    sign = -1.0;
                    laplacian_loss_sqrt = -laplacian_loss_sqrt;
                }

                float laplacian_loss_xyz[6];
                laplacian_loss_xyz[0] = 2.0 / (l_100 * (l_100 + l100)) * sign;
                laplacian_loss_xyz[1] = 2.0 / (l100 * (l_100 + l100)) * sign;
                float laplacian_loss_x = -(laplacian_loss_xyz[0] + laplacian_loss_xyz[1]);

                laplacian_loss_xyz[2] = 2.0 / (l_010 * (l_010 + l010)) * sign;
                laplacian_loss_xyz[3] = 2.0 / (l010 * (l_010 + l010)) * sign;
                float laplacian_loss_y = -(laplacian_loss_xyz[2] + laplacian_loss_xyz[3]);

                laplacian_loss_xyz[4] = 2.0 / (l_001 * (l_001 + l001)) * sign;
                laplacian_loss_xyz[5] = 2.0 / (l001 * (l_001 + l001)) * sign;
                float laplacian_loss_z = -(laplacian_loss_xyz[4] + laplacian_loss_xyz[5]);

                atomicAdd(&grad_data[c_index], (laplacian_loss_x + laplacian_loss_y + laplacian_loss_z) * scale);

                float ghost_b2a_coeff = 0.0;
                if (ghost_b_cor_index != -1)
                {
                    int dir_index = floor((double)ghost_b_cor_index / 2.0);

                    int pos_index[2];
                    if (dir_index == 0)
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 2;
                    }
                    else if (dir_index == 1)
                    {
                        pos_index[0] = 0;
                        pos_index[1] = 2;
                    }
                    else
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 0;
                    }

                    int g_i0 = node_gho_neighbors[c_index][dir_index][0];
                    int g_i1 = node_gho_neighbors[c_index][dir_index][1];
                    int g_i2 = node_gho_neighbors[c_index][dir_index][2];
                    int g_i3 = node_gho_neighbors[c_index][dir_index][3];

                    float c0 = node_gho_coeff[c_index][dir_index][0];
                    float c1 = node_gho_coeff[c_index][dir_index][1];
                    float c2 = node_gho_coeff[c_index][dir_index][2];
                    float c3 = node_gho_coeff[c_index][dir_index][3];

                    float laplacian_loss_sxyz = scale * laplacian_loss_xyz[ghost_b_cor_index];

                    atomicAdd(&grad_data[g_i0], laplacian_loss_sxyz * c0);
                    atomicAdd(&grad_data[g_i1], laplacian_loss_sxyz * c1);
                    atomicAdd(&grad_data[g_i2], laplacian_loss_sxyz * c2);
                    atomicAdd(&grad_data[g_i3], laplacian_loss_sxyz * c3);

                    int pos_index_all[4];
                    pos_index_all[0] = pos_index[0] * 2;
                    pos_index_all[1] = pos_index[0] * 2 + 1;
                    pos_index_all[2] = pos_index[1] * 2;
                    pos_index_all[3] = pos_index[1] * 2 + 1;
                    for (int i = 0; i < 4; ++i)
                    {
                        if (ghost_a_cor_index == pos_index_all[i])
                        {
                            ghost_b2a_coeff = node_gho_coeff[c_index][dir_index][i + 4];
                            continue;
                        }

                        int g_i = node_all_neighbors[c_index][pos_index_all[i]];
                        float ci = node_gho_coeff[c_index][dir_index][i + 4];
                        atomicAdd(&grad_data[g_i], laplacian_loss_sxyz * (-ci));
                    }

                    atomicAdd(&grad_data[c_index], laplacian_loss_sxyz * node_gho_coeff[c_index][dir_index][8]);
                }

                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[c_index][i] != -1)
                    {
                        atomicAdd(&grad_data[node_all_neighbors[c_index][i]], scale * laplacian_loss_xyz[i]);
                        continue;
                    }

                    if (ghost_b_cor_index == i)
                    {
                        continue;
                    }

                    int dir_index = floor((double)i / 2.0);
                    int g_i0 = node_gho_neighbors[c_index][dir_index][0];
                    int g_i1 = node_gho_neighbors[c_index][dir_index][1];

                    int pos_index = node_gho_neighbors[c_index][dir_index][2];
                    int g_i2 = node_all_neighbors[c_index][pos_index * 2];
                    int g_i3 = node_all_neighbors[c_index][pos_index * 2 + 1];

                    float c0 = node_gho_coeff[c_index][dir_index][0];
                    float c1 = node_gho_coeff[c_index][dir_index][1];
                    float c2 = node_gho_coeff[c_index][dir_index][2];
                    float c3 = node_gho_coeff[c_index][dir_index][3];
                    float c4 = node_gho_coeff[c_index][dir_index][4];

                    float laplacian_loss_sxyz = scale * laplacian_loss_xyz[i];

                    atomicAdd(&grad_data[g_i0], laplacian_loss_sxyz * c1);
                    atomicAdd(&grad_data[g_i1], laplacian_loss_sxyz * c0);
                    atomicAdd(&grad_data[g_i2], laplacian_loss_sxyz * (-c2));
                    atomicAdd(&grad_data[g_i3], laplacian_loss_sxyz * (-c3));
                    atomicAdd(&grad_data[c_index], laplacian_loss_sxyz * c4);

                    if (ghost_b_cor_index != -1 && ghost_a_cor_index == i)
                    {
                        laplacian_loss_sxyz = scale * laplacian_loss_xyz[ghost_b_cor_index];
                        float laplacian_loss_sxyz_bc = laplacian_loss_sxyz * (-ghost_b2a_coeff);

                        atomicAdd(&grad_data[g_i0], laplacian_loss_sxyz_bc * c1);
                        atomicAdd(&grad_data[g_i1], laplacian_loss_sxyz_bc * c0);
                        atomicAdd(&grad_data[g_i2], laplacian_loss_sxyz_bc * (-c2));
                        atomicAdd(&grad_data[g_i3], laplacian_loss_sxyz_bc * (-c3));
                        atomicAdd(&grad_data[c_index], laplacian_loss_sxyz_bc * c4);
                    }
                }

                atomicAdd(&total_laplacian_loss[0], laplacian_loss_sqrt);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void viscosity_loss_fused_kernel_LOT_R(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                float scale,
                float epsilon,
                int start_dim, int end_dim,
                size_t Q,
                float valid_threshold,
                float *__restrict__ total_eikonal_loss,
                float *__restrict__ total_viscosity_loss,
                float *__restrict__ valid_loss_size,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float l_100 = node_all_neighlen[c_index][0];
            const float l100 = node_all_neighlen[c_index][1];
            const float l_010 = node_all_neighlen[c_index][2];
            const float l010 = node_all_neighlen[c_index][3];
            const float l_001 = node_all_neighlen[c_index][4];
            const float l001 = node_all_neighlen[c_index][5];

            if (l_100 != 1.0 && l100 != 1.0 && l_010 != 1.0 && l010 != 1.0 && l_001 != 1.0 && l001 != 1.0)
            {
                const float v000 = sdf_data[c_index][idx];

                float v_value[6];
                int ghost_a_cor_index = -1;
                int ghost_b_cor_index = -1;
                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[c_index][i] != -1)
                    {
                        v_value[i] = sdf_data[node_all_neighbors[c_index][i]][idx];
                    }
                    else
                    {
                        int dir_index = floor((double)i / 2.0);
                        if (node_gho_neighbors[c_index][dir_index][3] == -1)
                        {
                            float v0 = sdf_data[node_gho_neighbors[c_index][dir_index][0]][idx];
                            float v1 = sdf_data[node_gho_neighbors[c_index][dir_index][1]][idx];

                            int pos_index = node_gho_neighbors[c_index][dir_index][2];
                            float v2 = sdf_data[node_all_neighbors[c_index][pos_index * 2]][idx];
                            float v3 = sdf_data[node_all_neighbors[c_index][pos_index * 2 + 1]][idx];

                            float c0 = node_gho_coeff[c_index][dir_index][0];
                            float c1 = node_gho_coeff[c_index][dir_index][1];
                            float c2 = node_gho_coeff[c_index][dir_index][2];
                            float c3 = node_gho_coeff[c_index][dir_index][3];
                            float c4 = node_gho_coeff[c_index][dir_index][4];

                            v_value[i] = c0 * v1 + c1 * v0 - c2 * v2 - c3 * v3 + c4 * v000;

                            ghost_a_cor_index = i;
                        }
                        else
                        {
                            ghost_b_cor_index = i;
                        }
                    }
                }

                if (ghost_b_cor_index != -1)
                {
                    int dir_index = floor((double)ghost_b_cor_index / 2.0);

                    int pos_index[2];
                    if (dir_index == 0)
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 2;
                    }
                    else if (dir_index == 1)
                    {
                        pos_index[0] = 0;
                        pos_index[1] = 2;
                    }
                    else
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 0;
                    }

                    float v0 = sdf_data[node_gho_neighbors[c_index][dir_index][0]][idx];
                    float v1 = sdf_data[node_gho_neighbors[c_index][dir_index][1]][idx];
                    float v2 = sdf_data[node_gho_neighbors[c_index][dir_index][2]][idx];
                    float v3 = sdf_data[node_gho_neighbors[c_index][dir_index][3]][idx];
                    float v4 = v_value[pos_index[0] * 2];
                    float v5 = v_value[pos_index[0] * 2 + 1];
                    float v6 = v_value[pos_index[1] * 2];
                    float v7 = v_value[pos_index[1] * 2 + 1];

                    float c0 = node_gho_coeff[c_index][dir_index][0];
                    float c1 = node_gho_coeff[c_index][dir_index][1];
                    float c2 = node_gho_coeff[c_index][dir_index][2];
                    float c3 = node_gho_coeff[c_index][dir_index][3];
                    float c4 = node_gho_coeff[c_index][dir_index][4];
                    float c5 = node_gho_coeff[c_index][dir_index][5];
                    float c6 = node_gho_coeff[c_index][dir_index][6];
                    float c7 = node_gho_coeff[c_index][dir_index][7];
                    float c8 = node_gho_coeff[c_index][dir_index][8];

                    v_value[ghost_b_cor_index] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3 - c4 * v4 - c5 * v5 - c6 * v6 - c7 * v7 + c8 * v000;
                }

                float x_length = l_100 + l100;
                float x0_dx_coeff = l100 / (l_100 * x_length);
                float x1_dx_coeff = l_100 / (l100 * x_length);
                float x_dx_coeff = x0_dx_coeff - x1_dx_coeff;
                float Dx = x1_dx_coeff * v_value[1] + x_dx_coeff * v000 - x0_dx_coeff * v_value[0];
                float x0_d2x_coeff = 2.0 / (l_100 * x_length);
                float x1_d2x_coeff = 2.0 / (l100 * x_length);
                float x_d2x_coeff = x0_d2x_coeff + x1_d2x_coeff;
                float D2x = x1_d2x_coeff * v_value[1] - x_d2x_coeff * v000 + x0_d2x_coeff * v_value[0];

                float y_length = l_010 + l010;
                float y0_dy_coeff = l010 / (l_010 * y_length);
                float y1_dy_coeff = l_010 / (l010 * y_length);
                float y_dy_coeff = y0_dy_coeff - y1_dy_coeff;
                float Dy = y1_dy_coeff * v_value[3] + y_dy_coeff * v000 - y0_dy_coeff * v_value[2];
                float y0_d2y_coeff = 2.0 / (l_010 * y_length);
                float y1_d2y_coeff = 2.0 / (l010 * y_length);
                float y_d2y_coeff = y0_d2y_coeff + y1_d2y_coeff;
                float D2y = y1_d2y_coeff * v_value[3] - y_d2y_coeff * v000 + y0_d2y_coeff * v_value[2];

                float z_length = l_001 + l001;
                float z0_dz_coeff = l001 / (l_001 * z_length);
                float z1_dz_coeff = l_001 / (l001 * z_length);
                float z_dz_coeff = z0_dz_coeff - z1_dz_coeff;
                float Dz = z1_dz_coeff * v_value[5] + z_dz_coeff * v000 - z0_dz_coeff * v_value[4];
                float z0_d2z_coeff = 2.0 / (l_001 * z_length);
                float z1_d2z_coeff = 2.0 / (l001 * z_length);
                float z_d2z_coeff = z0_d2z_coeff + z1_d2z_coeff;
                float D2z = z1_d2z_coeff * v_value[5] - z_d2z_coeff * v000 + z0_d2z_coeff * v_value[4];

                float sdf_grad_len = sqrt(Dx * Dx + Dy * Dy + Dz * Dz + 1e-8);
                float eikonal_loss = sdf_grad_len - 1.0;
                float sdf_sign = (v000 >= 0.0) ? 1.0 : -1.0;
                sdf_sign = (v000 == 0.0) ? 0.0 : sdf_sign;
                float viscosity_loss_sqrt = sdf_sign * eikonal_loss - epsilon * (D2x + D2y + D2z);
                float viscosity_loss = viscosity_loss_sqrt * viscosity_loss_sqrt;

                float sdf_grad_len_inv = 1.0 / sdf_grad_len;
                float viscosity_loss_0 = sdf_sign * sdf_grad_len_inv;

                float viscosity_los_xyz[6];
                viscosity_los_xyz[0] = viscosity_loss_0 * Dx * (-x0_dx_coeff) - epsilon * x0_d2x_coeff;
                float viscosity_loss_x = viscosity_loss_0 * Dx * x_dx_coeff - epsilon * (-x_d2x_coeff);
                viscosity_los_xyz[1] = viscosity_loss_0 * Dx * x1_dx_coeff - epsilon * x1_d2x_coeff;

                viscosity_los_xyz[2] = viscosity_loss_0 * Dy * (-y0_dy_coeff) - epsilon * y0_d2y_coeff;
                float viscosity_loss_y = viscosity_loss_0 * Dy * y_dy_coeff - epsilon * (-y_d2y_coeff);
                viscosity_los_xyz[3] = viscosity_loss_0 * Dy * y1_dy_coeff - epsilon * y1_d2y_coeff;

                viscosity_los_xyz[4] = viscosity_loss_0 * Dz * (-z0_dz_coeff) - epsilon * z0_d2z_coeff;
                float viscosity_loss_z = viscosity_loss_0 * Dz * z_dz_coeff - epsilon * (-z_d2z_coeff);
                viscosity_los_xyz[5] = viscosity_loss_0 * Dz * z1_dz_coeff - epsilon * z1_d2z_coeff;

                float viscosity_loss_scale = scale * 2.0 * viscosity_loss_sqrt;

                float ghost_b2a_coeff = 0.0;
                if (ghost_b_cor_index != -1)
                {
                    int dir_index = floor((double)ghost_b_cor_index / 2.0);

                    int pos_index[2];
                    if (dir_index == 0)
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 2;
                    }
                    else if (dir_index == 1)
                    {
                        pos_index[0] = 0;
                        pos_index[1] = 2;
                    }
                    else
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 0;
                    }

                    int g_i0 = node_gho_neighbors[c_index][dir_index][0];
                    int g_i1 = node_gho_neighbors[c_index][dir_index][1];
                    int g_i2 = node_gho_neighbors[c_index][dir_index][2];
                    int g_i3 = node_gho_neighbors[c_index][dir_index][3];

                    float c0 = node_gho_coeff[c_index][dir_index][0];
                    float c1 = node_gho_coeff[c_index][dir_index][1];
                    float c2 = node_gho_coeff[c_index][dir_index][2];
                    float c3 = node_gho_coeff[c_index][dir_index][3];

                    float viscosity_loss_sxyz = viscosity_loss_scale * viscosity_los_xyz[ghost_b_cor_index];

                    atomicAdd(&grad_data[g_i0], viscosity_loss_sxyz * c0);
                    atomicAdd(&grad_data[g_i1], viscosity_loss_sxyz * c1);
                    atomicAdd(&grad_data[g_i2], viscosity_loss_sxyz * c2);
                    atomicAdd(&grad_data[g_i3], viscosity_loss_sxyz * c3);

                    int pos_index_all[4];
                    pos_index_all[0] = pos_index[0] * 2;
                    pos_index_all[1] = pos_index[0] * 2 + 1;
                    pos_index_all[2] = pos_index[1] * 2;
                    pos_index_all[3] = pos_index[1] * 2 + 1;
                    for (int i = 0; i < 4; ++i)
                    {
                        if (ghost_a_cor_index == pos_index_all[i])
                        {
                            ghost_b2a_coeff = node_gho_coeff[c_index][dir_index][i + 4];
                            continue;
                        }

                        int g_i = node_all_neighbors[c_index][pos_index_all[i]];
                        float ci = node_gho_coeff[c_index][dir_index][i + 4];
                        atomicAdd(&grad_data[g_i], viscosity_loss_sxyz * (-ci));
                    }

                    atomicAdd(&grad_data[c_index], viscosity_loss_sxyz * node_gho_coeff[c_index][dir_index][8]);
                }

                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[c_index][i] != -1)
                    {
                        atomicAdd(&grad_data[node_all_neighbors[c_index][i]], viscosity_loss_scale * viscosity_los_xyz[i]);
                        continue;
                    }

                    if (ghost_b_cor_index == i)
                    {
                        continue;
                    }

                    int dir_index = floor((double)i / 2.0);
                    int g_i0 = node_gho_neighbors[c_index][dir_index][0];
                    int g_i1 = node_gho_neighbors[c_index][dir_index][1];

                    int pos_index = node_gho_neighbors[c_index][dir_index][2];
                    int g_i2 = node_all_neighbors[c_index][pos_index * 2];
                    int g_i3 = node_all_neighbors[c_index][pos_index * 2 + 1];

                    float c0 = node_gho_coeff[c_index][dir_index][0];
                    float c1 = node_gho_coeff[c_index][dir_index][1];
                    float c2 = node_gho_coeff[c_index][dir_index][2];
                    float c3 = node_gho_coeff[c_index][dir_index][3];
                    float c4 = node_gho_coeff[c_index][dir_index][4];

                    float viscosity_loss_sxyz = viscosity_loss_scale * viscosity_los_xyz[i];

                    atomicAdd(&grad_data[g_i0], viscosity_loss_sxyz * c1);
                    atomicAdd(&grad_data[g_i1], viscosity_loss_sxyz * c0);
                    atomicAdd(&grad_data[g_i2], viscosity_loss_sxyz * (-c2));
                    atomicAdd(&grad_data[g_i3], viscosity_loss_sxyz * (-c3));
                    atomicAdd(&grad_data[c_index], viscosity_loss_sxyz * c4);

                    if (ghost_b_cor_index != -1 && ghost_a_cor_index == i)
                    {
                        viscosity_loss_sxyz = viscosity_loss_scale * viscosity_los_xyz[ghost_b_cor_index];
                        float viscosity_loss_sxyz_bc = viscosity_loss_sxyz * (-ghost_b2a_coeff);

                        atomicAdd(&grad_data[g_i0], viscosity_loss_sxyz_bc * c1);
                        atomicAdd(&grad_data[g_i1], viscosity_loss_sxyz_bc * c0);
                        atomicAdd(&grad_data[g_i2], viscosity_loss_sxyz_bc * (-c2));
                        atomicAdd(&grad_data[g_i3], viscosity_loss_sxyz_bc * (-c3));
                        atomicAdd(&grad_data[c_index], viscosity_loss_sxyz_bc * c4);
                    }
                }

                float viscosity_loss_grad = (viscosity_loss_x + viscosity_loss_y + viscosity_loss_z) * viscosity_loss_scale;
                atomicAdd(&grad_data[c_index], viscosity_loss_grad);

                if (fabsf(v000) <= valid_threshold)
                {
                    atomicAdd(&total_eikonal_loss[0], eikonal_loss * eikonal_loss);
                    atomicAdd(&total_viscosity_loss[0], viscosity_loss);
                    atomicAdd(&valid_loss_size[0], 1.0);
                }

                /*float x_multi = l_100 * l100 * (l_100 + l100);
                float l_100_2 = l_100 * l_100;
                float l100_2 = l100 * l100;
                float Dx = (-l100_2 * v_value[0] + (l100_2 - l_100_2) * l100 * v000 + l_100_2 * v_value[1]) / x_multi;
                float D2x = (2.0 * l100 * v_value[0] - 2.0 * (l_100 + l100) * v000 + 2.0 * l_100 * v_value[1]) / x_multi;

                float y_multi = l_010 * l010 * (l_010 + l010);
                float l_010_2 = l_010 * l_010;
                float l010_2 = l010 * l010;
                float Dy = (-l010_2 * v_value[2] + (l010_2 - l_010_2) * l010 * v000 + l_010_2 * v_value[3]) / y_multi;
                float D2y = (2.0 * l010 * v_value[2] - 2.0 * (l_010 + l010) * v000 + 2.0 * l_010 * v_value[3]) / y_multi;

                float z_multi = l_001 * l001 * (l_001 + l001);
                float l_001_2 = l_001 * l_001;
                float l001_2 = l001 * l001;
                float Dz = (-l001_2 * v_value[4] + (l001_2 - l_001_2) * l001 * v000 + l_001_2 * v_value[5]) / z_multi;
                float D2z = (2.0 * l001 * v_value[4] - 2.0 * (l_001 + l001) * v000 + 2.0 * l_001 * v_value[5]) / z_multi;

                float sdf_grad_len = sqrt(Dx * Dx + Dy * Dy + Dz * Dz + 1e-8);
                float eikonal_loss = sdf_grad_len - 1.0;
                float sdf_sign = (v000 >= 0.0) ? 1.0 : -1.0;
                sdf_sign = (v000 == 0.0) ? 0.0 : sdf_sign;
                float viscosity_loss_sqrt = sdf_sign * eikonal_loss - epsilon * (D2x + D2y + D2z);
                float viscosity_loss = viscosity_loss_sqrt * viscosity_loss_sqrt;

                float sdf_grad_len_inv = 1.0 / sdf_grad_len;
                float viscosity_loss_0 = sdf_sign * sdf_grad_len_inv;

                float viscosity_los_xyz[6];
                viscosity_los_xyz[0] = viscosity_loss_0 * Dx * (-l100_2 / x_multi) - epsilon * (2.0 * l100 / x_multi);
                float viscosity_loss_x = viscosity_loss_0 * Dx * ((l100_2 - l_100_2) * l100 / x_multi) - epsilon * (-2.0 * (l_100 + l100) / x_multi);
                viscosity_los_xyz[1] = viscosity_loss_0 * Dx * (l_100_2 / x_multi) - epsilon * (2.0 * l_100 / x_multi);

                viscosity_los_xyz[2] = viscosity_loss_0 * Dy * (-l010_2 / y_multi) - epsilon * (2.0 * l010 / y_multi);
                float viscosity_loss_y = viscosity_loss_0 * Dy * ((l010_2 - l_010_2) * l010 / y_multi) - epsilon * (-2.0 * (l_010 + l010) / y_multi);
                viscosity_los_xyz[3] = viscosity_loss_0 * Dy * (l_010_2 / y_multi) - epsilon * (2.0 * l_010 / y_multi);

                viscosity_los_xyz[4] = viscosity_loss_0 * Dz * (-l001_2 / z_multi) - epsilon * (2.0 * l001 / z_multi);
                float viscosity_loss_z = viscosity_loss_0 * Dz * ((l001_2 - l_001_2) * l001 / z_multi) - epsilon * (-2.0 * (l_001 + l001) / z_multi);
                viscosity_los_xyz[5] = viscosity_loss_0 * Dz * (l_001_2 / z_multi) - epsilon * (2.0 * l_001 / z_multi);

                float viscosity_loss_scale = scale * 2.0 * viscosity_loss_sqrt;*/
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void gaussian_sdf_conv_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_gauss_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_gauss_kernals,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                const int32_t *__restrict__ geo_corner_map,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ gauss_sdf_out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            int g_index = geo_corner_map[c_index];

            int32_t corner_index = node_gauss_neighbors[g_index][idx];

            if (corner_index >= 0)
            {
                atomicAdd(&gauss_sdf_out[c_index], node_gauss_kernals[g_index][idx] * sdf_data[corner_index][0]);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void gaussian_gradient_conv_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_gauss_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_gauss_kernals,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> gradient_data,
                const int32_t *__restrict__ corner_indices,
                const int32_t *__restrict__ geo_corner_map,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ gauss_gradient_out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            int g_index = geo_corner_map[c_index];

            int32_t corner_index = node_gauss_neighbors[g_index][idx];

            if (corner_index >= 0)
            {
                int32_t g_corner_index = geo_corner_map[corner_index];

                atomicAdd(&gauss_gradient_out[g_index * 3], node_gauss_kernals[g_index][idx] * gradient_data[g_corner_index][0]);
                atomicAdd(&gauss_gradient_out[g_index * 3 + 1], node_gauss_kernals[g_index][idx] * gradient_data[g_corner_index][1]);
                atomicAdd(&gauss_gradient_out[g_index * 3 + 2], node_gauss_kernals[g_index][idx] * gradient_data[g_corner_index][2]);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void gaussian_sdf_conv_backward_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_gauss_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_gauss_kernals,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> grad_data,
                const int32_t *__restrict__ corner_indices,
                const int32_t *__restrict__ geo_corner_map,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ gauss_grad_out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const int g_index = geo_corner_map[c_index];

            int32_t corner_index = node_gauss_neighbors[g_index][idx];

            if (corner_index >= 0)
            {
                atomicAdd(&gauss_grad_out[corner_index], node_gauss_kernals[g_index][idx] * grad_data[c_index][0]);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void com_corner_gradient_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                const int32_t *__restrict__ geo_corner_map,
                size_t Q,
                float *__restrict__ gradient_out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);

            const int c_index = corner_indices[tid];

            int g_index = geo_corner_map[c_index];

            const float l_100 = node_all_neighlen[g_index][0];
            const float l100 = node_all_neighlen[g_index][1];
            const float l_010 = node_all_neighlen[g_index][2];
            const float l010 = node_all_neighlen[g_index][3];
            const float l_001 = node_all_neighlen[g_index][4];
            const float l001 = node_all_neighlen[g_index][5];

            if (l_100 != 1.0 && l100 != 1.0 && l_010 != 1.0 && l010 != 1.0 && l_001 != 1.0 && l001 != 1.0)
            {
                const float v000 = sdf_data[c_index][0];

                float v_value[6];
                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[g_index][i] != -1)
                    {
                        v_value[i] = sdf_data[node_all_neighbors[g_index][i]][0];
                    }
                    else
                    {
                        int dir_index = floor((double)i / 2.0);
                        if (node_gho_neighbors[g_index][dir_index][3] == -1)
                        {
                            float v0 = sdf_data[node_gho_neighbors[g_index][dir_index][0]][0];
                            float v1 = sdf_data[node_gho_neighbors[g_index][dir_index][1]][0];

                            float c0 = node_gho_coeff[g_index][dir_index][0];
                            float c1 = node_gho_coeff[g_index][dir_index][1];

                            // printf("c_c0= %f, c_c1= %f\n", c0, c1);

                            v_value[i] = c0 * v1 + c1 * v0;
                        }
                        else
                        {
                            float v0 = sdf_data[node_gho_neighbors[g_index][dir_index][0]][0];
                            float v1 = sdf_data[node_gho_neighbors[g_index][dir_index][1]][0];
                            float v2 = sdf_data[node_gho_neighbors[g_index][dir_index][2]][0];
                            float v3 = sdf_data[node_gho_neighbors[g_index][dir_index][3]][0];

                            float c0 = node_gho_coeff[g_index][dir_index][0];
                            float c1 = node_gho_coeff[g_index][dir_index][1];
                            float c2 = node_gho_coeff[g_index][dir_index][2];
                            float c3 = node_gho_coeff[g_index][dir_index][3];

                            // printf("cc_c0= %f, cc_c1= %f, cc_c2= %f, cc_c3= %f\n", c0, c1, c2, c3);

                            v_value[i] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3;
                        }
                    }
                }

                float Dx = ((v_value[1] - v000) / l100) * (l_100 / (l_100 + l100)) + ((v000 - v_value[0]) / l_100) * (l100 / (l_100 + l100));
                float Dy = ((v_value[3] - v000) / l010) * (l_010 / (l_010 + l010)) + ((v000 - v_value[2]) / l_010) * (l010 / (l_010 + l010));
                float Dz = ((v_value[5] - v000) / l001) * (l_001 / (l_001 + l001)) + ((v000 - v_value[4]) / l_001) * (l001 / (l_001 + l001));

                atomicAdd(&gradient_out[g_index * 3], Dx);
                atomicAdd(&gradient_out[g_index * 3 + 1], Dy);
                atomicAdd(&gradient_out[g_index * 3 + 2], Dz);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void com_corner_gradient_thirdord_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                size_t Q,
                float *__restrict__ gradient_out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);

            const int c_index = corner_indices[tid];

            const float l_100 = node_all_neighlen[c_index][0];
            const float l100 = node_all_neighlen[c_index][1];
            const float l_010 = node_all_neighlen[c_index][2];
            const float l010 = node_all_neighlen[c_index][3];
            const float l_001 = node_all_neighlen[c_index][4];
            const float l001 = node_all_neighlen[c_index][5];

            if (l_100 != 1.0 && l100 != 1.0 && l_010 != 1.0 && l010 != 1.0 && l_001 != 1.0 && l001 != 1.0)
            {
                const float v000 = sdf_data[c_index][0];

                float v_value[6];

                int ghost_b_cor_index = -1;
                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[c_index][i] != -1)
                    {
                        v_value[i] = sdf_data[node_all_neighbors[c_index][i]][0];
                    }
                    else
                    {
                        int dir_index = floor((double)i / 2.0);
                        if (node_gho_neighbors[c_index][dir_index][3] == -1)
                        {
                            float v0 = sdf_data[node_gho_neighbors[c_index][dir_index][0]][0];
                            float v1 = sdf_data[node_gho_neighbors[c_index][dir_index][1]][0];

                            int pos_index = node_gho_neighbors[c_index][dir_index][2];
                            float v2 = sdf_data[node_all_neighbors[c_index][pos_index * 2]][0];
                            float v3 = sdf_data[node_all_neighbors[c_index][pos_index * 2 + 1]][0];

                            float c0 = node_gho_coeff[c_index][dir_index][0];
                            float c1 = node_gho_coeff[c_index][dir_index][1];
                            float c2 = node_gho_coeff[c_index][dir_index][2];
                            float c3 = node_gho_coeff[c_index][dir_index][3];
                            float c4 = node_gho_coeff[c_index][dir_index][4];

                            v_value[i] = c0 * v1 + c1 * v0 - c2 * v2 - c3 * v3 + c4 * v000;
                        }
                        else
                        {
                            ghost_b_cor_index = i;
                        }
                    }
                }

                if (ghost_b_cor_index != -1)
                {
                    int dir_index = floor((double)ghost_b_cor_index / 2.0);

                    int pos_index[2];
                    if (dir_index == 0)
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 2;
                    }
                    else if (dir_index == 1)
                    {
                        pos_index[0] = 0;
                        pos_index[1] = 2;
                    }
                    else
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 0;
                    }

                    float v0 = sdf_data[node_gho_neighbors[c_index][dir_index][0]][0];
                    float v1 = sdf_data[node_gho_neighbors[c_index][dir_index][1]][0];
                    float v2 = sdf_data[node_gho_neighbors[c_index][dir_index][2]][0];
                    float v3 = sdf_data[node_gho_neighbors[c_index][dir_index][3]][0];
                    float v4 = v_value[pos_index[0] * 2];
                    float v5 = v_value[pos_index[0] * 2 + 1];
                    float v6 = v_value[pos_index[1] * 2];
                    float v7 = v_value[pos_index[1] * 2 + 1];

                    float c0 = node_gho_coeff[c_index][dir_index][0];
                    float c1 = node_gho_coeff[c_index][dir_index][1];
                    float c2 = node_gho_coeff[c_index][dir_index][2];
                    float c3 = node_gho_coeff[c_index][dir_index][3];
                    float c4 = node_gho_coeff[c_index][dir_index][4];
                    float c5 = node_gho_coeff[c_index][dir_index][5];
                    float c6 = node_gho_coeff[c_index][dir_index][6];
                    float c7 = node_gho_coeff[c_index][dir_index][7];
                    float c8 = node_gho_coeff[c_index][dir_index][8];

                    v_value[ghost_b_cor_index] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3 - c4 * v4 - c5 * v5 - c6 * v6 - c7 * v7 + c8 * v000;
                    // v_value[ghost_b_cor_index] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3;
                }

                float Dx = ((v_value[1] - v000) / l100) * (l_100 / (l_100 + l100)) + ((v000 - v_value[0]) / l_100) * (l100 / (l_100 + l100));
                float Dy = ((v_value[3] - v000) / l010) * (l_010 / (l_010 + l010)) + ((v000 - v_value[2]) / l_010) * (l010 / (l_010 + l010));
                float Dz = ((v_value[5] - v000) / l001) * (l_001 / (l_001 + l001)) + ((v000 - v_value[4]) / l_001) * (l001 / (l_001 + l001));

                atomicAdd(&gradient_out[c_index * 3], Dx);
                atomicAdd(&gradient_out[c_index * 3 + 1], Dy);
                atomicAdd(&gradient_out[c_index * 3 + 2], Dz);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void gauss_gradient_smooth_fused_kernal_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_gauss_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_gauss_kernals,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> gauss_grad_diff_data,
                const int32_t *__restrict__ corner_indices,
                const int32_t *__restrict__ geo_corner_map,
                float scale,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ smooth_loss_out,
                float *__restrict__ grad_out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            int g_index = geo_corner_map[c_index];

            int32_t corner_index = node_gauss_neighbors[g_index][idx];
            if (corner_index >= 0)
            {
                int32_t g_corner_index = geo_corner_map[corner_index];

                const float l_100 = node_all_neighlen[g_corner_index][0];
                const float l100 = node_all_neighlen[g_corner_index][1];
                const float l_010 = node_all_neighlen[g_corner_index][2];
                const float l010 = node_all_neighlen[g_corner_index][3];
                const float l_001 = node_all_neighlen[g_corner_index][4];
                const float l001 = node_all_neighlen[g_corner_index][5];

                if (l_100 != 1.0 && l100 != 1.0 && l_010 != 1.0 && l010 != 1.0 && l_001 != 1.0 && l001 != 1.0)
                {
                    float gradient_gauss_diff_x_sqrt = gauss_grad_diff_data[g_index][0];
                    float gradient_gauss_diff_y_sqrt = gauss_grad_diff_data[g_index][1];
                    float gradient_gauss_diff_z_sqrt = gauss_grad_diff_data[g_index][2];

                    float dxyz_coeffs[6];

                    float xyz_grad_scale = scale * 2.0 * node_gauss_kernals[g_index][idx];

                    float x_grad_scale = xyz_grad_scale * gradient_gauss_diff_x_sqrt;
                    dxyz_coeffs[0] = (-l100 / (l_100 * (l_100 + l100))) * x_grad_scale;
                    dxyz_coeffs[1] = (l_100 / (l100 * (l_100 + l100))) * x_grad_scale;
                    float dx_coeff = -dxyz_coeffs[0] - dxyz_coeffs[1];

                    float y_grad_scale = xyz_grad_scale * gradient_gauss_diff_y_sqrt;
                    dxyz_coeffs[2] = (-l010 / (l_010 * (l_010 + l010))) * y_grad_scale;
                    dxyz_coeffs[3] = (l_010 / (l010 * (l_010 + l010))) * y_grad_scale;
                    float dy_coeff = -dxyz_coeffs[2] - dxyz_coeffs[3];

                    float z_grad_scale = xyz_grad_scale * gradient_gauss_diff_z_sqrt;
                    dxyz_coeffs[4] = (-l001 / (l_001 * (l_001 + l001))) * z_grad_scale;
                    dxyz_coeffs[5] = (l_001 / (l001 * (l_001 + l001))) * z_grad_scale;
                    float dz_coeff = -dxyz_coeffs[4] - dxyz_coeffs[5];

                    atomicAdd(&grad_out[corner_index], dx_coeff + dy_coeff + dz_coeff);

                    for (int i = 0; i < 6; ++i)
                    {
                        if (node_all_neighbors[g_corner_index][i] != -1)
                        {
                            atomicAdd(&grad_out[node_all_neighbors[g_corner_index][i]], dxyz_coeffs[i]);
                            continue;
                        }

                        int dir_index = floor((double)i / 2.0);

                        if (node_gho_neighbors[g_corner_index][dir_index][3] == -1)
                        {
                            int g_i0 = node_gho_neighbors[g_corner_index][dir_index][0];
                            int g_i1 = node_gho_neighbors[g_corner_index][dir_index][1];

                            float c0 = node_gho_coeff[g_corner_index][dir_index][0];
                            float c1 = node_gho_coeff[g_corner_index][dir_index][1];

                            // printf("c0= %f, c1= %f\n", c0, c1);

                            atomicAdd(&grad_out[g_i0], dxyz_coeffs[i] * c1);
                            atomicAdd(&grad_out[g_i1], dxyz_coeffs[i] * c0);
                        }
                        else
                        {
                            int g_i0 = node_gho_neighbors[g_corner_index][dir_index][0];
                            int g_i1 = node_gho_neighbors[g_corner_index][dir_index][1];
                            int g_i2 = node_gho_neighbors[g_corner_index][dir_index][2];
                            int g_i3 = node_gho_neighbors[g_corner_index][dir_index][3];

                            float c0 = node_gho_coeff[g_corner_index][dir_index][0];
                            float c1 = node_gho_coeff[g_corner_index][dir_index][1];
                            float c2 = node_gho_coeff[g_corner_index][dir_index][2];
                            float c3 = node_gho_coeff[g_corner_index][dir_index][3];

                            atomicAdd(&grad_out[g_i0], dxyz_coeffs[i] * c0);
                            atomicAdd(&grad_out[g_i1], dxyz_coeffs[i] * c1);
                            atomicAdd(&grad_out[g_i2], dxyz_coeffs[i] * c2);
                            atomicAdd(&grad_out[g_i3], dxyz_coeffs[i] * c3);
                        }
                    }

                    float c_smooth_loss = gradient_gauss_diff_x_sqrt * gradient_gauss_diff_x_sqrt +
                                          gradient_gauss_diff_y_sqrt * gradient_gauss_diff_y_sqrt +
                                          gradient_gauss_diff_z_sqrt * gradient_gauss_diff_z_sqrt;
                    atomicAdd(&smooth_loss_out[g_index], c_smooth_loss);
                }
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void gauss_gradient_smooth_fused_thirdord_kernal_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_gauss_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_gauss_kernals,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> gauss_grad_diff_data,
                const int32_t *__restrict__ corner_indices,
                float scale,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ smooth_loss_out,
                float *__restrict__ grad_out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            int32_t corner_index = node_gauss_neighbors[c_index][idx];
            if (corner_index >= 0)
            {
                const float l_100 = node_all_neighlen[corner_index][0];
                const float l100 = node_all_neighlen[corner_index][1];
                const float l_010 = node_all_neighlen[corner_index][2];
                const float l010 = node_all_neighlen[corner_index][3];
                const float l_001 = node_all_neighlen[corner_index][4];
                const float l001 = node_all_neighlen[corner_index][5];

                if (l_100 != 1.0 && l100 != 1.0 && l_010 != 1.0 && l010 != 1.0 && l_001 != 1.0 && l001 != 1.0)
                {
                    float gradient_gauss_diff_x_sqrt = gauss_grad_diff_data[c_index][0];
                    float gradient_gauss_diff_y_sqrt = gauss_grad_diff_data[c_index][1];
                    float gradient_gauss_diff_z_sqrt = gauss_grad_diff_data[c_index][2];

                    float dxyz_coeffs[6];

                    float xyz_grad_scale = scale * 2.0 * node_gauss_kernals[c_index][idx];

                    float x_grad_scale = xyz_grad_scale * gradient_gauss_diff_x_sqrt;
                    dxyz_coeffs[0] = (-l100 / (l_100 * (l_100 + l100))) * x_grad_scale;
                    dxyz_coeffs[1] = (l_100 / (l100 * (l_100 + l100))) * x_grad_scale;
                    float dx_coeff = -dxyz_coeffs[0] - dxyz_coeffs[1];

                    float y_grad_scale = xyz_grad_scale * gradient_gauss_diff_y_sqrt;
                    dxyz_coeffs[2] = (-l010 / (l_010 * (l_010 + l010))) * y_grad_scale;
                    dxyz_coeffs[3] = (l_010 / (l010 * (l_010 + l010))) * y_grad_scale;
                    float dy_coeff = -dxyz_coeffs[2] - dxyz_coeffs[3];

                    float z_grad_scale = xyz_grad_scale * gradient_gauss_diff_z_sqrt;
                    dxyz_coeffs[4] = (-l001 / (l_001 * (l_001 + l001))) * z_grad_scale;
                    dxyz_coeffs[5] = (l_001 / (l001 * (l_001 + l001))) * z_grad_scale;
                    float dz_coeff = -dxyz_coeffs[4] - dxyz_coeffs[5];

                    atomicAdd(&grad_out[corner_index], dx_coeff + dy_coeff + dz_coeff);

                    int ghost_a_cor_index = -1;
                    int ghost_b_cor_index = -1;
                    for (int i = 0; i < 6; ++i)
                    {
                        if (node_all_neighbors[corner_index][i] == -1)
                        {
                            int dir_index = floor((double)i / 2.0);

                            if (node_gho_neighbors[corner_index][dir_index][3] == -1)
                            {
                                ghost_a_cor_index = i;
                            }
                            else
                            {
                                ghost_b_cor_index = i;
                            }
                        }
                    }

                    float ghost_b2a_coeff = 0.0;
                    if (ghost_b_cor_index != -1)
                    {
                        int dir_index = floor((double)ghost_b_cor_index / 2.0);

                        int pos_index[2];
                        if (dir_index == 0)
                        {
                            pos_index[0] = 1;
                            pos_index[1] = 2;
                        }
                        else if (dir_index == 1)
                        {
                            pos_index[0] = 0;
                            pos_index[1] = 2;
                        }
                        else
                        {
                            pos_index[0] = 1;
                            pos_index[1] = 0;
                        }

                        int g_i0 = node_gho_neighbors[corner_index][dir_index][0];
                        int g_i1 = node_gho_neighbors[corner_index][dir_index][1];
                        int g_i2 = node_gho_neighbors[corner_index][dir_index][2];
                        int g_i3 = node_gho_neighbors[corner_index][dir_index][3];

                        float c0 = node_gho_coeff[corner_index][dir_index][0];
                        float c1 = node_gho_coeff[corner_index][dir_index][1];
                        float c2 = node_gho_coeff[corner_index][dir_index][2];
                        float c3 = node_gho_coeff[corner_index][dir_index][3];

                        atomicAdd(&grad_out[g_i0], dxyz_coeffs[ghost_b_cor_index] * c0);
                        atomicAdd(&grad_out[g_i1], dxyz_coeffs[ghost_b_cor_index] * c1);
                        atomicAdd(&grad_out[g_i2], dxyz_coeffs[ghost_b_cor_index] * c2);
                        atomicAdd(&grad_out[g_i3], dxyz_coeffs[ghost_b_cor_index] * c3);

                        int pos_index_all[4];
                        pos_index_all[0] = pos_index[0] * 2;
                        pos_index_all[1] = pos_index[0] * 2 + 1;
                        pos_index_all[2] = pos_index[1] * 2;
                        pos_index_all[3] = pos_index[1] * 2 + 1;
                        for (int i = 0; i < 4; ++i)
                        {
                            if (ghost_a_cor_index == pos_index_all[i])
                            {
                                ghost_b2a_coeff = node_gho_coeff[corner_index][dir_index][i + 4];
                                continue;
                            }

                            int g_i = node_all_neighbors[corner_index][pos_index_all[i]];
                            float ci = node_gho_coeff[corner_index][dir_index][i + 4];
                            atomicAdd(&grad_out[g_i], dxyz_coeffs[ghost_b_cor_index] * (-ci));
                        }

                        atomicAdd(&grad_out[corner_index], dxyz_coeffs[ghost_b_cor_index] * node_gho_coeff[corner_index][dir_index][8]);
                    }

                    for (int i = 0; i < 6; ++i)
                    {
                        if (node_all_neighbors[corner_index][i] != -1)
                        {
                            atomicAdd(&grad_out[node_all_neighbors[corner_index][i]], dxyz_coeffs[i]);
                            continue;
                        }

                        if (ghost_b_cor_index == i)
                        {
                            continue;
                        }

                        int dir_index = floor((double)i / 2.0);
                        int g_i0 = node_gho_neighbors[corner_index][dir_index][0];
                        int g_i1 = node_gho_neighbors[corner_index][dir_index][1];

                        int pos_index = node_gho_neighbors[corner_index][dir_index][2];
                        int g_i2 = node_all_neighbors[corner_index][pos_index * 2];
                        int g_i3 = node_all_neighbors[corner_index][pos_index * 2 + 1];

                        float c0 = node_gho_coeff[corner_index][dir_index][0];
                        float c1 = node_gho_coeff[corner_index][dir_index][1];
                        float c2 = node_gho_coeff[corner_index][dir_index][2];
                        float c3 = node_gho_coeff[corner_index][dir_index][3];
                        float c4 = node_gho_coeff[corner_index][dir_index][4];

                        /*if(c0 == -1.0)
                        {
                            printf("corner_index= %d, i= %d\n", corner_index, i);
                        }

                        printf("c0= %f, c1= %f, c2= %f, c3= %f, c4= %f\n", c0, c1, c2, c3, c4);*/

                        atomicAdd(&grad_out[g_i0], dxyz_coeffs[i] * c1);
                        atomicAdd(&grad_out[g_i1], dxyz_coeffs[i] * c0);
                        atomicAdd(&grad_out[g_i2], dxyz_coeffs[i] * (-c2));
                        atomicAdd(&grad_out[g_i3], dxyz_coeffs[i] * (-c3));
                        atomicAdd(&grad_out[corner_index], dxyz_coeffs[i] * c4);

                        if (ghost_b_cor_index != -1 && ghost_a_cor_index == i)
                        {
                            float dxyz_coeffs_bc = dxyz_coeffs[ghost_b_cor_index] * (-ghost_b2a_coeff);

                            atomicAdd(&grad_out[g_i0], dxyz_coeffs_bc * c1);
                            atomicAdd(&grad_out[g_i1], dxyz_coeffs_bc * c0);
                            atomicAdd(&grad_out[g_i2], dxyz_coeffs_bc * (-c2));
                            atomicAdd(&grad_out[g_i3], dxyz_coeffs_bc * (-c3));
                            atomicAdd(&grad_out[corner_index], dxyz_coeffs_bc * c4);
                        }
                    }

                    float c_smooth_loss = gradient_gauss_diff_x_sqrt * gradient_gauss_diff_x_sqrt +
                                          gradient_gauss_diff_y_sqrt * gradient_gauss_diff_y_sqrt +
                                          gradient_gauss_diff_z_sqrt * gradient_gauss_diff_z_sqrt;
                    atomicAdd(&smooth_loss_out[c_index], c_smooth_loss);
                }
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void viscosity_loss_fused_kernel_LOT_T1(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> node_gho_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_gho_coeff,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                float scale,
                float epsilon,
                int start_dim, int end_dim,
                size_t Q,
                float valid_threshold,
                float *__restrict__ total_eikonal_loss,
                float *__restrict__ total_viscosity_loss,
                float *__restrict__ valid_loss_size,
                float *__restrict__ dx_out,
                float *__restrict__ dy_out,
                float *__restrict__ dz_out,
                float *__restrict__ dxyz_out)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float l_100 = node_all_neighlen[c_index][0];
            const float l100 = node_all_neighlen[c_index][1];
            const float l_010 = node_all_neighlen[c_index][2];
            const float l010 = node_all_neighlen[c_index][3];
            const float l_001 = node_all_neighlen[c_index][4];
            const float l001 = node_all_neighlen[c_index][5];

            if (l_100 != 1.0 && l100 != 1.0 && l_010 != 1.0 && l010 != 1.0 && l_001 != 1.0 && l001 != 1.0)
            {
                const float v000 = sdf_data[c_index][idx];

                float v_value[6];
                int ghost_a_cor_index = -1;
                int ghost_b_cor_index = -1;
                for (int i = 0; i < 6; ++i)
                {
                    if (node_all_neighbors[c_index][i] != -1)
                    {
                        v_value[i] = sdf_data[node_all_neighbors[c_index][i]][idx];
                    }
                    else
                    {
                        int dir_index = floor((double)i / 2.0);
                        if (node_gho_neighbors[c_index][dir_index][3] == -1)
                        {
                            float v0 = sdf_data[node_gho_neighbors[c_index][dir_index][0]][idx];
                            float v1 = sdf_data[node_gho_neighbors[c_index][dir_index][1]][idx];

                            int pos_index = node_gho_neighbors[c_index][dir_index][2];
                            float v2 = sdf_data[node_all_neighbors[c_index][pos_index * 2]][idx];
                            float v3 = sdf_data[node_all_neighbors[c_index][pos_index * 2 + 1]][idx];

                            float c0 = node_gho_coeff[c_index][dir_index][0];
                            float c1 = node_gho_coeff[c_index][dir_index][1];
                            float c2 = node_gho_coeff[c_index][dir_index][2];
                            float c3 = node_gho_coeff[c_index][dir_index][3];
                            float c4 = node_gho_coeff[c_index][dir_index][4];

                            v_value[i] = c0 * v1 + c1 * v0 - c2 * v2 - c3 * v3 + c4 * v000;

                            ghost_a_cor_index = i;
                        }
                        else
                        {
                            ghost_b_cor_index = i;
                        }
                    }
                }

                if (ghost_b_cor_index != -1)
                {
                    int dir_index = floor((double)ghost_b_cor_index / 2.0);

                    int pos_index[2];
                    if (dir_index == 0)
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 2;
                    }
                    else if (dir_index == 1)
                    {
                        pos_index[0] = 0;
                        pos_index[1] = 2;
                    }
                    else
                    {
                        pos_index[0] = 1;
                        pos_index[1] = 0;
                    }

                    float v0 = sdf_data[node_gho_neighbors[c_index][dir_index][0]][idx];
                    float v1 = sdf_data[node_gho_neighbors[c_index][dir_index][1]][idx];
                    float v2 = sdf_data[node_gho_neighbors[c_index][dir_index][2]][idx];
                    float v3 = sdf_data[node_gho_neighbors[c_index][dir_index][3]][idx];
                    float v4 = v_value[pos_index[0] * 2];
                    float v5 = v_value[pos_index[0] * 2 + 1];
                    float v6 = v_value[pos_index[1] * 2];
                    float v7 = v_value[pos_index[1] * 2 + 1];

                    float c0 = node_gho_coeff[c_index][dir_index][0];
                    float c1 = node_gho_coeff[c_index][dir_index][1];
                    float c2 = node_gho_coeff[c_index][dir_index][2];
                    float c3 = node_gho_coeff[c_index][dir_index][3];
                    float c4 = node_gho_coeff[c_index][dir_index][4];
                    float c5 = node_gho_coeff[c_index][dir_index][5];
                    float c6 = node_gho_coeff[c_index][dir_index][6];
                    float c7 = node_gho_coeff[c_index][dir_index][7];
                    float c8 = node_gho_coeff[c_index][dir_index][8];

                    v_value[ghost_b_cor_index] = c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3 - c4 * v4 - c5 * v5 - c6 * v6 - c7 * v7 + c8 * v000;

                    /*if (c_index == 127)
                    {

                        printf("g_b_i= %d\n", ghost_b_cor_index);
                        printf("g_b= %f\n", v_value[ghost_b_cor_index]);
                        printf("v0= %f\n", v0);
                        printf("v1= %f\n", v1);
                        printf("v2= %f\n", v2);
                        printf("v3= %f\n", v3);
                        printf("v4= %f\n", v4);
                        printf("v5= %f\n", v5);
                        printf("v6= %f\n", v6);
                        printf("v7= %f\n", v7);
                    }*/
                }

                float x_multi = l_100 * l100 * (l_100 + l100);
                float l_100_2 = l_100 * l_100;
                float l100_2 = l100 * l100;
                // float Dx = (-l100_2 * v_value[0] + (l100_2 - l_100_2) * l100 * v000 + l_100_2 * v_value[1]) / x_multi;
                float Dx = ((v_value[1] - v000) / l100) * (l_100 / (l_100 + l100)) + ((v000 - v_value[0]) / l_100) * (l100 / (l_100 + l100));
                float D2x = (2.0 * l100 * v_value[0] - 2.0 * (l_100 + l100) * v000 + 2.0 * l_100 * v_value[1]) / x_multi;

                float y_multi = l_010 * l010 * (l_010 + l010);
                float l_010_2 = l_010 * l_010;
                float l010_2 = l010 * l010;
                // float Dy = (-l010_2 * v_value[2] + (l010_2 - l_010_2) * l010 * v000 + l_010_2 * v_value[3]) / y_multi;
                float Dy = ((v_value[3] - v000) / l010) * (l_010 / (l_010 + l010)) + ((v000 - v_value[2]) / l_010) * (l010 / (l_010 + l010));
                float D2y = (2.0 * l010 * v_value[2] - 2.0 * (l_010 + l010) * v000 + 2.0 * l_010 * v_value[3]) / y_multi;

                float z_multi = l_001 * l001 * (l_001 + l001);
                float l_001_2 = l_001 * l_001;
                float l001_2 = l001 * l001;
                // float Dz = (-l001_2 * v_value[4] + (l001_2 - l_001_2) * l001 * v000 + l_001_2 * v_value[5]) / z_multi;
                float Dz = ((v_value[5] - v000) / l001) * (l_001 / (l_001 + l001)) + ((v000 - v_value[4]) / l_001) * (l001 / (l_001 + l001));
                float D2z = (2.0 * l001 * v_value[4] - 2.0 * (l_001 + l001) * v000 + 2.0 * l_001 * v_value[5]) / z_multi;

                float sdf_grad_len = sqrt(Dx * Dx + Dy * Dy + Dz * Dz + 1e-8) - 1;

                if (c_index == 128)
                {
                    printf("sdf_grad_len= %f\n", sdf_grad_len);

                    printf("l_100= %f\n", l_100);
                    printf("l100= %f\n", l100);
                    printf("l_010= %f\n", l_010);
                    printf("l010= %f\n", l010);
                    printf("l_001= %f\n", l_001);
                    printf("l001= %f\n", l001);

                    printf("Dx= %f\n", Dx);
                    printf("Dy= %f\n", Dy);
                    printf("Dz= %f\n", Dz);
                    printf("D2x= %f\n", D2x);
                    printf("D2y= %f\n", D2y);
                    printf("D2z= %f\n", D2z);
                    printf("v0= %f\n", v_value[0]);
                    printf("v1= %f\n", v_value[1]);
                    printf("v2= %f\n", v_value[2]);
                    printf("v3= %f\n", v_value[3]);
                    printf("v4= %f\n", v_value[4]);
                    printf("v5= %f\n", v_value[5]);
                }

                atomicAdd(&dx_out[c_index], Dx);
                atomicAdd(&dy_out[c_index], Dy);
                atomicAdd(&dz_out[c_index], Dz);
                atomicAdd(&dxyz_out[c_index], sdf_grad_len);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void viscosity_loss_fused_kernel_LOT_S(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> node_all_neighlen,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                float scale,
                float epsilon,
                int start_dim, int end_dim,
                size_t Q,
                float valid_threshold,
                float *__restrict__ total_eikonal_loss,
                float *__restrict__ total_viscosity_loss,
                float *__restrict__ valid_loss_size,
                float *__restrict__ grad_data)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float l_100 = (node_all_neighbors[c_index][0] != -1) ? node_all_neighlen[c_index][0] : 0.f;
            const float l100 = (node_all_neighbors[c_index][1] != -1) ? node_all_neighlen[c_index][1] : 0.f;
            const float l_010 = (node_all_neighbors[c_index][2] != -1) ? node_all_neighlen[c_index][2] : 0.f;
            const float l010 = (node_all_neighbors[c_index][3] != -1) ? node_all_neighlen[c_index][3] : 0.f;
            const float l_001 = (node_all_neighbors[c_index][4] != -1) ? node_all_neighlen[c_index][4] : 0.f;
            const float l001 = (node_all_neighbors[c_index][5] != -1) ? node_all_neighlen[c_index][5] : 0.f;

            if (l_100 != 0.0 && l100 != 0.0 && l_010 != 0.0 && l010 != 0.0 && l_001 != 0.0 && l001 != 0.0)
            {
                const float v000 = sdf_data[c_index][idx];

                const float v_100 = sdf_data[node_all_neighbors[c_index][0]][idx];
                const float v100 = sdf_data[node_all_neighbors[c_index][1]][idx];
                const float v_010 = sdf_data[node_all_neighbors[c_index][2]][idx];
                const float v010 = sdf_data[node_all_neighbors[c_index][3]][idx];
                const float v_001 = sdf_data[node_all_neighbors[c_index][4]][idx];
                const float v001 = sdf_data[node_all_neighbors[c_index][5]][idx];

                float x_multi = l_100 * l100 * (l_100 + l100);
                float l_100_2 = l_100 * l_100;
                float l100_2 = l100 * l100;
                float Dx = (-l100_2 * v_100 + (l100_2 - l_100_2) * l100 * v000 + l_100_2 * v100) / x_multi;
                float D2x = (2.0 * l100 * v_100 - 2.0 * (l_100 + l100) * v000 + 2.0 * l_100 * v100) / x_multi;

                float y_multi = l_010 * l010 * (l_010 + l010);
                float l_010_2 = l_010 * l_010;
                float l010_2 = l010 * l010;
                float Dy = (-l010_2 * v_010 + (l010_2 - l_010_2) * l010 * v000 + l_010_2 * v010) / y_multi;
                float D2y = (2.0 * l010 * v_010 - 2.0 * (l_010 + l010) * v000 + 2.0 * l_010 * v010) / y_multi;

                float z_multi = l_001 * l001 * (l_001 + l001);
                float l_001_2 = l_001 * l_001;
                float l001_2 = l001 * l001;
                float Dz = (-l001_2 * v_001 + (l001_2 - l_001_2) * l001 * v000 + l_001_2 * v001) / z_multi;
                float D2z = (2.0 * l001 * v_001 - 2.0 * (l_001 + l001) * v000 + 2.0 * l_001 * v001) / z_multi;

                float sdf_grad_len = sqrt(Dx * Dx + Dy * Dy + Dz * Dz + 1e-8);
                float eikonal_loss = sdf_grad_len - 1.0;
                float sdf_sign = (v000 >= 0.0) ? 1.0 : -1.0;
                sdf_sign = (v000 == 0.0) ? 0.0 : sdf_sign;
                float viscosity_loss_sqrt = sdf_sign * eikonal_loss - epsilon * (D2x + D2y + D2z);
                float viscosity_loss = viscosity_loss_sqrt * viscosity_loss_sqrt;

                float sdf_grad_len_inv = 1.0 / sdf_grad_len;
                float viscosity_loss_0 = sdf_sign * sdf_grad_len_inv;

                float viscosity_loss_x0 = viscosity_loss_0 * Dx * (-l100_2 / x_multi) - epsilon * (2.0 * l100 / x_multi);
                float viscosity_loss_x = viscosity_loss_0 * Dx * ((l100_2 - l_100_2) * l100 / x_multi) - epsilon * (-2.0 * (l_100 + l100) / x_multi);
                float viscosity_loss_x1 = viscosity_loss_0 * Dx * (l_100_2 / x_multi) - epsilon * (2.0 * l_100 / x_multi);

                float viscosity_loss_y0 = viscosity_loss_0 * Dy * (-l010_2 / y_multi) - epsilon * (2.0 * l010 / y_multi);
                float viscosity_loss_y = viscosity_loss_0 * Dy * ((l010_2 - l_010_2) * l010 / y_multi) - epsilon * (-2.0 * (l_010 + l010) / y_multi);
                float viscosity_loss_y1 = viscosity_loss_0 * Dy * (l_010_2 / y_multi) - epsilon * (2.0 * l_010 / y_multi);

                float viscosity_loss_z0 = viscosity_loss_0 * Dz * (-l001_2 / z_multi) - epsilon * (2.0 * l001 / z_multi);
                float viscosity_loss_z = viscosity_loss_0 * Dz * ((l001_2 - l_001_2) * l001 / z_multi) - epsilon * (-2.0 * (l_001 + l001) / z_multi);
                float viscosity_loss_z1 = viscosity_loss_0 * Dz * (l_001_2 / z_multi) - epsilon * (2.0 * l_001 / z_multi);

                float viscosity_loss_scale = scale * 2.0 * viscosity_loss_sqrt;

                float viscosity_loss_grad = (viscosity_loss_x + viscosity_loss_y + viscosity_loss_z) * viscosity_loss_scale;

                atomicAdd(&grad_data[c_index], viscosity_loss_grad);
                atomicAdd(&grad_data[node_all_neighbors[c_index][0]], viscosity_loss_x0 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][1]], viscosity_loss_x1 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][2]], viscosity_loss_y0 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][3]], viscosity_loss_y1 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][4]], viscosity_loss_z0 * viscosity_loss_scale);
                atomicAdd(&grad_data[node_all_neighbors[c_index][5]], viscosity_loss_z1 * viscosity_loss_scale);

                if (fabsf(v000) <= valid_threshold)
                {
                    atomicAdd(&total_eikonal_loss[0], eikonal_loss * eikonal_loss);
                    atomicAdd(&total_viscosity_loss[0], viscosity_loss);
                    atomicAdd(&valid_loss_size[0], 1.0);
                }
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void thin_plate_kernel_LOT(
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_all_neighbors,
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> node_dc_neighbors,
                const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> sdf_data,
                const int32_t *__restrict__ corner_indices,
                float scaling,
                int start_dim, int end_dim,
                size_t Q,
                float *__restrict__ sdf_grad,
                float *__restrict__ total)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int idx = tid % (end_dim - start_dim) + start_dim;
            const int c_index = corner_indices[tid / (end_dim - start_dim)];

            const float v000 = sdf_data[c_index][idx];

            const float v_100 = (node_all_neighbors[c_index][0] != -1) ? sdf_data[node_all_neighbors[c_index][0]][idx] : 0.f;
            const float v100 = (node_all_neighbors[c_index][1] != -1) ? sdf_data[node_all_neighbors[c_index][1]][idx] : 0.f;
            const float v_010 = (node_all_neighbors[c_index][2] != -1) ? sdf_data[node_all_neighbors[c_index][2]][idx] : 0.f;
            const float v010 = (node_all_neighbors[c_index][3] != -1) ? sdf_data[node_all_neighbors[c_index][3]][idx] : 0.f;
            const float v_001 = (node_all_neighbors[c_index][4] != -1) ? sdf_data[node_all_neighbors[c_index][4]][idx] : 0.f;
            const float v001 = (node_all_neighbors[c_index][5] != -1) ? sdf_data[node_all_neighbors[c_index][5]][idx] : 0.f;

            if ((node_all_neighbors[c_index][0] != -1) && (node_all_neighbors[c_index][1] != -1))
            {
                float fxx = scaling * (v100 - 2.0 * v000 + v_100);

                total[c_index] += ((fxx / scaling) * (fxx / scaling));

                atomicAdd(&sdf_grad[c_index], -4.0 * fxx);
                atomicAdd(&sdf_grad[node_all_neighbors[c_index][0]], 2.0 * fxx);
                atomicAdd(&sdf_grad[node_all_neighbors[c_index][1]], 2.0 * fxx);
            }

            if ((node_all_neighbors[c_index][2] != -1) && (node_all_neighbors[c_index][3] != -1))
            {
                float fyy = scaling * (v010 - 2.0 * v000 + v_010);

                total[c_index] += ((fyy / scaling) * (fyy / scaling));

                atomicAdd(&sdf_grad[c_index], -4.0 * fyy);
                atomicAdd(&sdf_grad[node_all_neighbors[c_index][2]], 2.0 * fyy);
                atomicAdd(&sdf_grad[node_all_neighbors[c_index][3]], 2.0 * fyy);
            }

            if ((node_all_neighbors[c_index][4] != -1) && (node_all_neighbors[c_index][5] != -1))
            {
                float fzz = scaling * (v001 - 2.0 * v000 + v_001);

                total[c_index] += ((fzz / scaling) * (fzz / scaling));

                atomicAdd(&sdf_grad[c_index], -4.0 * fzz);
                atomicAdd(&sdf_grad[node_all_neighbors[c_index][2]], 2.0 * fzz);
                atomicAdd(&sdf_grad[node_all_neighbors[c_index][3]], 2.0 * fzz);
            }

            const float vx0y0 = (node_dc_neighbors[c_index][0] != -1) ? sdf_data[node_dc_neighbors[c_index][0]][idx] : 0.f;
            const float vx0y1 = (node_dc_neighbors[c_index][1] != -1) ? sdf_data[node_dc_neighbors[c_index][1]][idx] : 0.f;
            const float vx0z0 = (node_dc_neighbors[c_index][2] != -1) ? sdf_data[node_dc_neighbors[c_index][2]][idx] : 0.f;
            const float vx0z1 = (node_dc_neighbors[c_index][3] != -1) ? sdf_data[node_dc_neighbors[c_index][3]][idx] : 0.f;
            const float vy0z0 = (node_dc_neighbors[c_index][4] != -1) ? sdf_data[node_dc_neighbors[c_index][4]][idx] : 0.f;
            const float vy0z1 = (node_dc_neighbors[c_index][5] != -1) ? sdf_data[node_dc_neighbors[c_index][5]][idx] : 0.f;
            const float vx1y0 = (node_dc_neighbors[c_index][6] != -1) ? sdf_data[node_dc_neighbors[c_index][6]][idx] : 0.f;
            const float vx1y1 = (node_dc_neighbors[c_index][7] != -1) ? sdf_data[node_dc_neighbors[c_index][7]][idx] : 0.f;
            const float vx1z0 = (node_dc_neighbors[c_index][8] != -1) ? sdf_data[node_dc_neighbors[c_index][8]][idx] : 0.f;
            const float vx1z1 = (node_dc_neighbors[c_index][9] != -1) ? sdf_data[node_dc_neighbors[c_index][9]][idx] : 0.f;
            const float vy1z0 = (node_dc_neighbors[c_index][10] != -1) ? sdf_data[node_dc_neighbors[c_index][10]][idx] : 0.f;
            const float vy1z1 = (node_dc_neighbors[c_index][11] != -1) ? sdf_data[node_dc_neighbors[c_index][11]][idx] : 0.f;

            if ((node_dc_neighbors[c_index][7] != -1) && (node_dc_neighbors[c_index][0] != -1) &&
                (node_dc_neighbors[c_index][1] != -1) && (node_dc_neighbors[c_index][6] != -1))
            {
                float fxy = scaling * 0.25 * (vx1y1 + vx0y0 - vx0y1 - vx1y0);

                total[c_index] += (2 * (fxy / scaling) * (fxy / scaling));

                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][7]], fxy);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][0]], fxy);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][1]], -fxy);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][6]], -fxy);
            }

            if ((node_dc_neighbors[c_index][9] != -1) && (node_dc_neighbors[c_index][2] != -1) &&
                (node_dc_neighbors[c_index][3] != -1) && (node_dc_neighbors[c_index][8] != -1))
            {
                float fxz = scaling * 0.25 * (vx1z1 + vx0z0 - vx0z1 - vx1z0);

                total[c_index] += (2 * (fxz / scaling) * (fxz / scaling));

                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][9]], fxz);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][2]], fxz);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][3]], -fxz);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][8]], -fxz);
            }

            if ((node_dc_neighbors[c_index][11] != -1) && (node_dc_neighbors[c_index][4] != -1) &&
                (node_dc_neighbors[c_index][5] != -1) && (node_dc_neighbors[c_index][10] != -1))
            {
                float fyz = scaling * 0.25 * (vy1z1 + vy0z0 - vy0z1 - vy1z0);

                total[c_index] += (2 * (fyz / scaling) * (fyz / scaling));

                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][11]], fyz);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][4]], fyz);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][5]], -fyz);
                atomicAdd(&sdf_grad[node_dc_neighbors[c_index][10]], -fyz);
            }
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void msi_tv_grad_sparse_kernel(
                // (reso * 2, reso)
                const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> links,
                // (capacity, n_layers, n_channels)
                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> msi,
                const int32_t *__restrict__ rand_cells,
                float scale,
                float scale_last,
                size_t Q,
                // Output
                torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> msi_mask,
                torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_msi)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int MSI_DATA_DIM = msi.size(2);
            const int channel_id = tid % MSI_DATA_DIM;
            const int msi_idx = rand_cells[tid / MSI_DATA_DIM];

            const int z = msi_idx % msi.size(1);
            int tmp = msi_idx / msi.size(1);

            const int y = tmp % links.size(1);
            const int x = tmp / links.size(1);

            const int nx = (x == links.size(0) - 1) ? 0 : x + 1;
            const int ny = (y == links.size(1) - 1) ? 0 : y + 1;

            const int lnk00 = links[x][y];
            const int lnk01 = links[x][ny];
            const int lnk10 = links[nx][y];

            const float v00 = lnk00 >= 0 ? msi[lnk00][z][channel_id] : 0.f;
            const float v_nxl = (lnk00 >= 0 && z + 1 < msi.size(1)) ? msi[lnk00][z + 1][channel_id] : ((channel_id == MSI_DATA_DIM - 1) ? 0.f : v00);
            const float v01 = lnk01 >= 0 ? msi[lnk01][z][channel_id] : 0.f;
            const float v10 = lnk10 >= 0 ? msi[lnk10][z][channel_id] : 0.f;

            if (channel_id == MSI_DATA_DIM - 1)
            {
                scale = scale_last;
            }

            float dx = (v10 - v00);
            float dy = (v01 - v00);
            float dz = (v_nxl - v00);
            const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);
            // printf("x=%d y=%d z=%d nx=%d ny=%d dx=%f dy=%f dz=%f scale=%f\n", x, y, z,
            //        nx, ny, dx, dy, dz, scale);

            // const float msi_nlayers = msi.size(1);

            // const float radius = msi_nlayers / (msi_nlayers - z - 0.5f);
            // const float nxl_radius = msi_nlayers / (msi_nlayers - z - 1.5f);
            // const float invr = 1.f / radius;
            // float coord00[3], coord01[3], coord10[3];
            // _equirect2unitvec(x, y, links.size(1), coord00);
            // _equirect2unitvec(x, ny, links.size(1), coord01);
            // _equirect2unitvec(nx, y, links.size(1), coord10);
            // printf("r=%f nlr=%f coord00[%f %f %f] coord01[%f %f %f] coord10[%f %f %f]\n",
            //         radius, nxl_radius,
            //         coord00[0], coord00[1], coord00[2],
            //         coord01[0], coord01[1], coord01[2],
            //         coord10[0], coord10[1], coord10[2]);

            // xsuby3d(coord01, coord00);
            // xsuby3d(coord10, coord00);
            // dx *= _rnorm(coord10) * invr;
            // dy *= _rnorm(coord01) * invr;
            // dz *= 1.f / (nxl_radius - radius);
            dx *= links.size(0) * (1.f / 256.f);
            dy *= links.size(1) * (1.f / 256.f);
            dz *= msi.size(1) * (1.f / 256.f);

#define MAYBE_ADD_SET(link, zz, val)                             \
    if (link >= 0 && val != 0.f)                                 \
    {                                                            \
        atomicAdd(&grad_msi[link][zz][channel_id], val *idelta); \
        if (msi_mask.size(0) > 0)                                \
            msi_mask[link][zz] = true;                           \
    }

            const float sm = -(dx + dy + dz);
            MAYBE_ADD_SET(lnk00, z, sm);
            if (z + 1 < msi.size(1))
            {
                MAYBE_ADD_SET(lnk00, z + 1, dz);
            }
            MAYBE_ADD_SET(lnk01, z, dy);
            MAYBE_ADD_SET(lnk10, z, dx);
#undef MAYBE_ADD_SET
        }

        __launch_bounds__(TV_GRAD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
            __global__ void lumisphere_tv_grad_sparse_kernel(
                const PackedSparseGridSpec grid,
                const int32_t *__restrict__ rand_cells,
                const float *__restrict__ sphfunc_val,
                const float *__restrict__ sphfunc_val_u,
                float scale,
                size_t Q,
                float ndc_coeffx,
                float ndc_coeffy,
                float dir_factor,
                // Output
                PackedGridOutputGrads grads)
        {
            CUDA_GET_THREAD_ID_U64(tid, Q);
            const int lane_id = tid & 0x1F;
            if (lane_id >= grid.sh_data_dim)
                return;
            const int point_id = tid >> 5;
            const int point_blk_id = threadIdx.x >> 5;

            const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
            const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

            const int idx = lane_id;

            const int xyz = rand_cells[point_id];
            const int z = xyz % (grid.size[2] - 1);
            const int xy = xyz / (grid.size[2] - 1);
            const int y = xy % (grid.size[1] - 1);
            const int x = xy / (grid.size[1] - 1);

            // __shared__ float grad_sphfunc_val[TV_GRAD_POINTS_PER_BLOCK][10];
            // __shared__ float grad_sphfunc_val_u[TV_GRAD_POINTS_PER_BLOCK][10];
            __shared__ typename WarpReducef::TempStorage temp_storage[TV_GRAD_POINTS_PER_BLOCK];

            uint32_t use_mask = (1U << grid.sh_data_dim) - 1;

            // Currently, will not work for MLP
            __syncwarp(use_mask);

            const int32_t *__restrict__ links_ptr = grid.links +
                                                    (x * grid.stride_x + y * grid.size[2] + z);

            if (*links_ptr == 0)
                return;

            float scaling[3];
            CALCULATE_RAY_SCALE(scaling, grid.size[0], grid.size[1], grid.size[2]);

            const int offx = grid.stride_x, offy = grid.size[2];

            const float v000 = links_ptr[0] >= 0 ? grid.sh_data[links_ptr[0] * grid.sh_data_dim + idx] : 0.f;
            const float v001 = links_ptr[1] >= 0 ? grid.sh_data[links_ptr[1] * grid.sh_data_dim + idx] : v000,
                        v010 = links_ptr[offy] >= 0 ? grid.sh_data[links_ptr[offy] * grid.sh_data_dim + idx] : v000,
                        v100 = links_ptr[offx] >= 0 ? grid.sh_data[links_ptr[offx] * grid.sh_data_dim + idx] : v000;

            const float sv = sphfunc_val[lane_colorgrp_id];
            const float v000a = v000 * sv,
                        v001a = v001 * sv,
                        v010a = v010 * sv,
                        v100a = v100 * sv;
            const float v000u = v000 * sphfunc_val_u[lane_colorgrp_id];

            const bool is_leader = lane_colorgrp_id == 0;
            float v000a_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(v000a, is_leader);
            float v001a_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(v001a, is_leader);
            float v010a_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(v010a, is_leader);
            float v100a_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(v100a, is_leader);
            float v000u_sum = WarpReducef(temp_storage[point_blk_id]).HeadSegmentedSum(v000u, is_leader);

            const float scale_u = dir_factor;

            float dx = (v100a_sum - v000a_sum) * scaling[0];
            float dy = (v010a_sum - v000a_sum) * scaling[1];
            float dz = (v001a_sum - v000a_sum) * scaling[2];
            float du = (v000u_sum - v000a_sum) * scale_u;

            int leader_id = lane_colorgrp * grid.basis_dim;
            dx = __shfl_sync(use_mask, dx, leader_id);
            dy = __shfl_sync(use_mask, dy, leader_id);
            dz = __shfl_sync(use_mask, dz, leader_id);
            du = __shfl_sync(use_mask, du, leader_id);

            const float idelta = scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz + du * du);

            dx *= scaling[0];
            dy *= scaling[1];
            dz *= scaling[2];
            du *= scale_u;

#define MAYBE_ADD_SET(gp, val)                                                              \
    if (links_ptr[gp] >= 0 && val != 0.f)                                                   \
    {                                                                                       \
        atomicAdd(&grads.grad_sh_out[links_ptr[gp] * grid.sh_data_dim + idx], val *idelta); \
        if (grads.mask_out != nullptr)                                                      \
        {                                                                                   \
            grads.mask_out[links_ptr[gp]] = true;                                           \
        }                                                                                   \
    }

            const float sm = -dx * sv - dy * sv - dz * sv +
                             du * (sphfunc_val_u[lane_colorgrp_id] - sv);
            MAYBE_ADD_SET(0, sm);
            MAYBE_ADD_SET(1, dz * sv);
            MAYBE_ADD_SET(offy, dy * sv);
            MAYBE_ADD_SET(offx, dx * sv);

#undef MAYBE_ADD_SET

            // TODO
            // __syncwarp(use_mask);
            // if (lane_id < grid.basis_dim) {
            //     calc_sphfunc_backward(
            //             grid,
            //             lane_id,
            //             point_id,
            //             dir,
            //             sphfunc_val[point_blk_id],
            //             grad_sphfunc_val_v[point_blk_id],
            //             grad_basis_out);
            //     calc_sphfunc_backward(
            //             grid,
            //             lane_id,
            //             point_id,
            //             dir_u,
            //             sphfunc_val_u[point_blk_id],
            //             grad_sphfunc_val[point_blk_id],
            //             grad_basis_out);
            //     calc_sphfunc_backward(
            //             grid,
            //             lane_id,
            //             point_id,
            //             dir_v,
            //             sphfunc_val_v[point_blk_id],
            //             grad_sphfunc_val_v[point_blk_id],
            //             grad_basis_out);
            // }
        }

    } // namespace device
} // namespace

torch::Tensor tv(torch::Tensor links, torch::Tensor data,
                 int start_dim, int end_dim,
                 bool use_logalpha,
                 float logalpha_delta,
                 bool ignore_edge,
                 float ndc_coeffx,
                 float ndc_coeffy)
{
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(links);
    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(!links.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);

    int nl = (links.size(0) - 1) * (links.size(1) - 1) * (links.size(2) - 1);
    size_t Q = nl * size_t(end_dim - start_dim);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TV_GRAD_CUDA_THREADS);
    torch::Tensor result = torch::zeros({}, data.options());
    device::tv_kernel<<<blocks, TV_GRAD_CUDA_THREADS>>>(
        links.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        start_dim,
        end_dim,
        1.f / nl,
        Q,
        ignore_edge,
        ndc_coeffx, ndc_coeffy,
        // Output
        result.data_ptr<float>());
    CUDA_CHECK_ERRORS;
    return result;
}

void tv_grad(torch::Tensor links,
             torch::Tensor data,
             int start_dim, int end_dim,
             float scale,
             bool use_logalpha,
             float logalpha_delta,
             bool ignore_edge,
             float ndc_coeffx,
             float ndc_coeffy,
             torch::Tensor grad_data)
{
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(links);
    CHECK_INPUT(grad_data);
    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!links.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);
    TORCH_CHECK(grad_data.ndimension() == 2);

    int nl = (links.size(0) - 1) * (links.size(1) - 1) * (links.size(2) - 1);
    size_t Q = nl * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::tv_grad_kernel<<<blocks, cuda_n_threads>>>(
        links.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        start_dim,
        end_dim,
        scale / nl,
        Q,
        ignore_edge,
        ndc_coeffx, ndc_coeffy,
        // Output
        grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void tv_grad_sparse(torch::Tensor links,
                    torch::Tensor data,
                    torch::Tensor rand_cells,
                    torch::Tensor mask_out,
                    int start_dim, int end_dim,
                    float scale,
                    bool use_logalpha,
                    float logalpha_delta,
                    bool ignore_edge,
                    bool ignore_last_z,
                    float ndc_coeffx,
                    float ndc_coeffy,
                    torch::Tensor grad_data)
{
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(links);
    CHECK_INPUT(grad_data);
    CHECK_INPUT(rand_cells);
    CHECK_INPUT(mask_out);
    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!links.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);
    TORCH_CHECK(grad_data.ndimension() == 2);

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::tv_grad_sparse_kernel<<<blocks, cuda_n_threads>>>(
        links.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        rand_cells.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        scale / nl,
        Q,
        ignore_edge,
        ignore_last_z,
        ndc_coeffx, ndc_coeffy,
        // Output
        (mask_out.dim() > 0) ? mask_out.data_ptr<bool>() : nullptr,
        grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void tv_grad_sparse_LOT(torch::Tensor node_neighs,
                        torch::Tensor data,
                        torch::Tensor rand_cells,
                        int start_dim, int end_dim,
                        float scale,
                        bool use_logalpha,
                        torch::Tensor grad_data)
{
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(node_neighs);
    CHECK_INPUT(grad_data);
    CHECK_INPUT(rand_cells);

    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_neighs.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(node_neighs.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::tv_grad_sparse_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_neighs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        rand_cells.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        scale / nl,
        Q,
        grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void tv_grad_sparse_thirdord_LOT(torch::Tensor node_all_neighbors,
                                 torch::Tensor node_all_neighlen,
                                 torch::Tensor node_gho_neighbors,
                                 torch::Tensor node_gho_coeff,
                                 torch::Tensor geo_corner_map,
                                 torch::Tensor data,
                                 torch::Tensor rand_cells,
                                 int start_dim, int end_dim,
                                 float scale,
                                 torch::Tensor grad_data)
{
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(grad_data);
    CHECK_INPUT(rand_cells);

    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::tv_grad_sparse_thirdord_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        rand_cells.data_ptr<int32_t>(),
        geo_corner_map.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        scale / nl,
        Q,
        grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void tv_grad_sparse_thirdord_mid_LOT(torch::Tensor node_all_neighbors,
                                     torch::Tensor node_all_neighlen,
                                     torch::Tensor node_gho_neighbors,
                                     torch::Tensor node_gho_coeff,
                                     torch::Tensor geo_corner_map,
                                     torch::Tensor data,
                                     torch::Tensor rand_cells,
                                     int start_dim, int end_dim,
                                     float scale,
                                     torch::Tensor grad_data)
{
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(grad_data);
    CHECK_INPUT(rand_cells);

    TORCH_CHECK(data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::tv_grad_sparse_thirdord_mid_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        rand_cells.data_ptr<int32_t>(),
        geo_corner_map.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        scale / nl,
        Q,
        grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void sample_tri_interp(torch::Tensor node_corners,
                       torch::Tensor sample_pos,
                       torch::Tensor low_pos,
                       torch::Tensor node_length,
                       torch::Tensor data,
                       torch::Tensor rand_cells,
                       int start_dim, int end_dim,
                       torch::Tensor out_data)
{
    DEVICE_GUARD(data);
    CHECK_INPUT(data);

    int nl = sample_pos.size(0);
    size_t Q = sample_pos.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::sample_tri_interp_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_corners.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        sample_pos.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        low_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        node_length.data_ptr<float>(),
        rand_cells.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        Q,
        out_data.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void sample_tri_min_interp(torch::Tensor node_corners,
                           torch::Tensor sample_pos,
                           torch::Tensor low_pos,
                           torch::Tensor node_length,
                           torch::Tensor data,
                           torch::Tensor rand_cells,
                           float sdf_thresh, float sdf_offset,
                           bool is_abs,
                           int start_dim, int end_dim,
                           torch::Tensor out_data)
{
    DEVICE_GUARD(data);
    CHECK_INPUT(data);

    int nl = sample_pos.size(0);
    size_t Q = sample_pos.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::sample_tri_interp_min_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_corners.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        sample_pos.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        low_pos.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        node_length.data_ptr<float>(),
        rand_cells.data_ptr<int32_t>(),
        sdf_thresh,
        sdf_offset,
        is_abs,
        start_dim,
        end_dim,
        Q,
        out_data.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void find_index_LOT(torch::Tensor hash_index,
                    torch::Tensor hash_corner_num,
                    torch::Tensor index_tensor)
{
    DEVICE_GUARD(hash_index);
    CHECK_INPUT(hash_index);
    CHECK_INPUT(hash_corner_num);
    CHECK_INPUT(index_tensor);

    TORCH_CHECK(!hash_index.is_floating_point());
    TORCH_CHECK(!hash_corner_num.is_floating_point());
    TORCH_CHECK(!index_tensor.is_floating_point());
    TORCH_CHECK(hash_index.ndimension() == 1);
    TORCH_CHECK(hash_corner_num.ndimension() == 1);
    TORCH_CHECK(index_tensor.ndimension() == 1);

    size_t Q = hash_index.size(0);

    printf("g_b_i= %d\n", Q);
    thrust::device_vector<int32_t> d_hash_index(hash_index.data_ptr<int32_t>(), hash_index.data_ptr<int32_t>() + Q);
    printf("g_b_i= %d\n", Q + 1);
    thrust::inclusive_scan(thrust::host, d_hash_index.begin(), d_hash_index.end(), d_hash_index.begin());

    //     printf("g_b_i= %d\n", Q);
    //     thrust::device_vector<int32_t> d_hash_index(hash_index.data_ptr<int32_t>(), hash_index.data_ptr<int32_t>() + Q);
    //     printf("g_b_i= %d\n", Q+1);
    //     thrust::device_vector<int32_t> permutation(hash_corner_num.data_ptr<int32_t>(), hash_corner_num.data_ptr<int32_t>() + Q);
    //     printf("g_b_i= %d\n", Q+2);
    //     thrust::inclusive_scan_by_key(thrust::host, d_hash_index.begin(), d_hash_index.end(), permutation.begin(), permutation.begin());
    printf("g_b_i= %d\n", Q + 3);
    thrust::copy(d_hash_index.begin(), d_hash_index.end(), index_tensor.data_ptr<int32_t>());
    CUDA_CHECK_ERRORS;
}

void sdf_grad_LOT(torch::Tensor node_all_neighbors,
                  torch::Tensor node_all_neighlen,
                  torch::Tensor sdf_data,
                  torch::Tensor corner_indices,
                  int start_dim, int end_dim,
                  torch::Tensor sdf_grad)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(sdf_grad.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);
    TORCH_CHECK(sdf_grad.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::sdf_grad_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        Q,
        sdf_grad.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void uni_sdf_grad_LOT(torch::Tensor node_all_neighbors,
                      torch::Tensor node_all_neighlen,
                      torch::Tensor sdf_data,
                      torch::Tensor corner_indices,
                      int start_dim, int end_dim,
                      torch::Tensor sdf_grad)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(sdf_grad.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(sdf_grad.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::uni_sdf_grad_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        Q,
        sdf_grad.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void uni_sdf_grad_backward_LOT(torch::Tensor node_all_neighbors,
                               torch::Tensor node_all_neighlen,
                               torch::Tensor sdf_grad,
                               torch::Tensor corner_indices,
                               float scale,
                               int start_dim, int end_dim,
                               torch::Tensor grad_data)
{
    DEVICE_GUARD(sdf_grad);
    CHECK_INPUT(sdf_grad);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_grad.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(sdf_grad.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::uni_sdf_grad_backward_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sdf_grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scale / nl,
        start_dim,
        end_dim,
        Q,
        grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void uni_viscosity_loss_fused_LOT(torch::Tensor node_all_neighbors,
                                  torch::Tensor node_all_neighlen,
                                  torch::Tensor sdf_data,
                                  torch::Tensor corner_indices,
                                  float scale,
                                  float epsilon,
                                  int start_dim, int end_dim,
                                  torch::Tensor total_eikonal_loss,
                                  torch::Tensor total_viscosity_loss,
                                  torch::Tensor grad_data)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::uni_viscosity_loss_fused_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scale / nl,
        epsilon,
        start_dim,
        end_dim,
        Q,
        total_eikonal_loss.data_ptr<float>(),
        total_viscosity_loss.data_ptr<float>(),
        grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void uni_laplacian_loss_fused_LOT(torch::Tensor node_all_neighbors,
                                  torch::Tensor node_all_neighlen,
                                  torch::Tensor sdf_data,
                                  torch::Tensor corner_indices,
                                  float scale,
                                  float epsilon,
                                  int start_dim, int end_dim,
                                  torch::Tensor total_laplacian_loss,
                                  torch::Tensor grad_data)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::uni_laplacian_loss_fused_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scale / nl,
        epsilon,
        start_dim,
        end_dim,
        Q,
        total_laplacian_loss.data_ptr<float>(),
        grad_data.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void laplacian_loss_fused_LOT(torch::Tensor node_all_neighbors,
                              torch::Tensor node_all_neighlen,
                              torch::Tensor node_gho_neighbors,
                              torch::Tensor node_gho_coeff,
                              torch::Tensor sdf_data,
                              torch::Tensor corner_indices,
                              float scale,
                              int start_dim, int end_dim,
                              torch::Tensor total_laplacian_loss,
                              torch::Tensor grad_data)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(!node_gho_neighbors.is_floating_point());
    TORCH_CHECK(node_gho_coeff.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);
    TORCH_CHECK(node_gho_neighbors.ndimension() == 3);
    TORCH_CHECK(node_gho_coeff.ndimension() == 3);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::laplacian_loss_fused_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scale / nl,
        start_dim,
        end_dim,
        Q,
        total_laplacian_loss.data_ptr<float>(),
        grad_data.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void viscosity_loss_fused_LOT_R(torch::Tensor node_all_neighbors,
                                torch::Tensor node_all_neighlen,
                                torch::Tensor node_gho_neighbors,
                                torch::Tensor node_gho_coeff,
                                torch::Tensor sdf_data,
                                torch::Tensor corner_indices,
                                float scale,
                                float epsilon,
                                float valid_threshold,
                                int start_dim, int end_dim,
                                torch::Tensor total_eikonal_loss,
                                torch::Tensor total_viscosity_loss,
                                torch::Tensor valid_loss_size,
                                torch::Tensor grad_data)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(!node_gho_neighbors.is_floating_point());
    TORCH_CHECK(node_gho_coeff.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);
    TORCH_CHECK(node_gho_neighbors.ndimension() == 3);
    TORCH_CHECK(node_gho_coeff.ndimension() == 3);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::viscosity_loss_fused_kernel_LOT_R<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scale / nl,
        epsilon,
        start_dim,
        end_dim,
        Q,
        valid_threshold,
        total_eikonal_loss.data_ptr<float>(),
        total_viscosity_loss.data_ptr<float>(),
        valid_loss_size.data_ptr<float>(),
        grad_data.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void gaussian_sdf_conv_LOT(torch::Tensor node_gauss_neighbors,
                           torch::Tensor node_gauss_kernals,
                           torch::Tensor geo_corner_map,
                           torch::Tensor sdf_data,
                           torch::Tensor corner_indices,
                           int start_dim, int end_dim,
                           torch::Tensor gauss_sdf_out)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_gauss_neighbors);
    CHECK_INPUT(node_gauss_kernals);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(gauss_sdf_out.is_floating_point());
    TORCH_CHECK(!node_gauss_neighbors.is_floating_point());
    TORCH_CHECK(node_gauss_kernals.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(gauss_sdf_out.ndimension() == 2);
    TORCH_CHECK(node_gauss_neighbors.ndimension() == 2);
    TORCH_CHECK(node_gauss_kernals.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::gaussian_sdf_conv_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_gauss_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gauss_kernals.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        geo_corner_map.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        Q,
        gauss_sdf_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void gaussian_gradient_conv_LOT(torch::Tensor node_gauss_neighbors,
                                torch::Tensor node_gauss_kernals,
                                torch::Tensor geo_corner_map,
                                torch::Tensor gradient_data,
                                torch::Tensor corner_indices,
                                int start_dim, int end_dim,
                                torch::Tensor gauss_gradient_out)
{
    DEVICE_GUARD(gradient_data);
    CHECK_INPUT(gradient_data);
    CHECK_INPUT(node_gauss_neighbors);
    CHECK_INPUT(node_gauss_kernals);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(gradient_data.is_floating_point());
    TORCH_CHECK(gauss_gradient_out.is_floating_point());
    TORCH_CHECK(!node_gauss_neighbors.is_floating_point());
    TORCH_CHECK(node_gauss_kernals.is_floating_point());
    TORCH_CHECK(gradient_data.ndimension() == 2);
    TORCH_CHECK(gauss_gradient_out.ndimension() == 2);
    TORCH_CHECK(node_gauss_neighbors.ndimension() == 2);
    TORCH_CHECK(node_gauss_kernals.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::gaussian_gradient_conv_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_gauss_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gauss_kernals.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        gradient_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        geo_corner_map.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        Q,
        gauss_gradient_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void gaussian_sdf_conv_backward_LOT(torch::Tensor node_gauss_neighbors,
                                    torch::Tensor node_gauss_kernals,
                                    torch::Tensor geo_corner_map,
                                    torch::Tensor grad_data,
                                    torch::Tensor corner_indices,
                                    int start_dim, int end_dim,
                                    torch::Tensor gauss_grad_out)
{
    DEVICE_GUARD(grad_data);
    CHECK_INPUT(grad_data);
    CHECK_INPUT(node_gauss_neighbors);
    CHECK_INPUT(node_gauss_kernals);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(gauss_grad_out.is_floating_point());
    TORCH_CHECK(!node_gauss_neighbors.is_floating_point());
    TORCH_CHECK(node_gauss_kernals.is_floating_point());
    TORCH_CHECK(grad_data.ndimension() == 2);
    TORCH_CHECK(gauss_grad_out.ndimension() == 2);
    TORCH_CHECK(node_gauss_neighbors.ndimension() == 2);
    TORCH_CHECK(node_gauss_kernals.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::gaussian_sdf_conv_backward_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_gauss_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gauss_kernals.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        geo_corner_map.data_ptr<int32_t>(),
        start_dim,
        end_dim,
        Q,
        gauss_grad_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void com_corner_gradient_LOT(torch::Tensor node_all_neighbors,
                             torch::Tensor node_all_neighlen,
                             torch::Tensor node_gho_neighbors,
                             torch::Tensor node_gho_coeff,
                             torch::Tensor geo_corner_map,
                             torch::Tensor sdf_data,
                             torch::Tensor corner_indices,
                             torch::Tensor gradient_out)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(node_gho_neighbors);
    CHECK_INPUT(node_gho_coeff);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(gradient_out.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(!node_gho_neighbors.is_floating_point());
    TORCH_CHECK(node_gho_coeff.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(gradient_out.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);
    TORCH_CHECK(node_gho_neighbors.ndimension() == 3);
    TORCH_CHECK(node_gho_coeff.ndimension() == 3);

    size_t Q = corner_indices.size(0);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::com_corner_gradient_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        geo_corner_map.data_ptr<int32_t>(),
        Q,
        gradient_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void com_corner_gradient_thirdord_LOT(torch::Tensor node_all_neighbors,
                                      torch::Tensor node_all_neighlen,
                                      torch::Tensor node_gho_neighbors,
                                      torch::Tensor node_gho_coeff,
                                      torch::Tensor sdf_data,
                                      torch::Tensor corner_indices,
                                      torch::Tensor gradient_out)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(node_gho_neighbors);
    CHECK_INPUT(node_gho_coeff);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(gradient_out.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(!node_gho_neighbors.is_floating_point());
    TORCH_CHECK(node_gho_coeff.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(gradient_out.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);
    TORCH_CHECK(node_gho_neighbors.ndimension() == 3);
    TORCH_CHECK(node_gho_coeff.ndimension() == 3);

    size_t Q = corner_indices.size(0);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::com_corner_gradient_thirdord_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        Q,
        gradient_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void gauss_gradient_smooth_fused_LOT(torch::Tensor node_all_neighbors,
                                     torch::Tensor node_gho_neighbors,
                                     torch::Tensor node_gauss_neighbors,
                                     torch::Tensor node_all_neighlen,
                                     torch::Tensor node_gho_coeff,
                                     torch::Tensor node_gauss_kernals,
                                     torch::Tensor geo_corner_map,
                                     torch::Tensor gauss_grad_diff_data,
                                     torch::Tensor corner_indices,
                                     float scale,
                                     int start_dim, int end_dim,
                                     torch::Tensor smooth_loss_out,
                                     torch::Tensor grad_out)
{
    DEVICE_GUARD(gauss_grad_diff_data);
    CHECK_INPUT(gauss_grad_diff_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_gho_neighbors);
    CHECK_INPUT(node_gauss_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(node_gho_coeff);
    CHECK_INPUT(node_gauss_kernals);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(gauss_grad_diff_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(!node_gho_neighbors.is_floating_point());
    TORCH_CHECK(node_gho_coeff.is_floating_point());
    TORCH_CHECK(!node_gauss_neighbors.is_floating_point());
    TORCH_CHECK(node_gauss_kernals.is_floating_point());
    TORCH_CHECK(gauss_grad_diff_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);
    TORCH_CHECK(node_gho_neighbors.ndimension() == 3);
    TORCH_CHECK(node_gho_coeff.ndimension() == 3);
    TORCH_CHECK(node_gauss_neighbors.ndimension() == 2);
    TORCH_CHECK(node_gauss_kernals.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::gauss_gradient_smooth_fused_kernal_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_gauss_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        node_gauss_kernals.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        gauss_grad_diff_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        geo_corner_map.data_ptr<int32_t>(),
        scale / nl,
        start_dim,
        end_dim,
        Q,
        smooth_loss_out.data_ptr<float>(),
        grad_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void gauss_gradient_smooth_fused_thirdord_LOT(torch::Tensor node_all_neighbors,
                                              torch::Tensor node_gho_neighbors,
                                              torch::Tensor node_gauss_neighbors,
                                              torch::Tensor node_all_neighlen,
                                              torch::Tensor node_gho_coeff,
                                              torch::Tensor node_gauss_kernals,
                                              torch::Tensor gauss_grad_diff_data,
                                              torch::Tensor corner_indices,
                                              float scale,
                                              int start_dim, int end_dim,
                                              torch::Tensor smooth_loss_out,
                                              torch::Tensor grad_out)
{
    DEVICE_GUARD(gauss_grad_diff_data);
    CHECK_INPUT(gauss_grad_diff_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_gho_neighbors);
    CHECK_INPUT(node_gauss_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(node_gho_coeff);
    CHECK_INPUT(node_gauss_kernals);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(gauss_grad_diff_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(!node_gho_neighbors.is_floating_point());
    TORCH_CHECK(node_gho_coeff.is_floating_point());
    TORCH_CHECK(!node_gauss_neighbors.is_floating_point());
    TORCH_CHECK(node_gauss_kernals.is_floating_point());
    TORCH_CHECK(gauss_grad_diff_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);
    TORCH_CHECK(node_gho_neighbors.ndimension() == 3);
    TORCH_CHECK(node_gho_coeff.ndimension() == 3);
    TORCH_CHECK(node_gauss_neighbors.ndimension() == 2);
    TORCH_CHECK(node_gauss_kernals.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::gauss_gradient_smooth_fused_thirdord_kernal_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_gauss_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        node_gauss_kernals.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        gauss_grad_diff_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scale / nl,
        start_dim,
        end_dim,
        Q,
        smooth_loss_out.data_ptr<float>(),
        grad_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void viscosity_loss_fused_LOT_T1(torch::Tensor node_all_neighbors,
                                 torch::Tensor node_all_neighlen,
                                 torch::Tensor node_gho_neighbors,
                                 torch::Tensor node_gho_coeff,
                                 torch::Tensor sdf_data,
                                 torch::Tensor corner_indices,
                                 float scale,
                                 float epsilon,
                                 float valid_threshold,
                                 int start_dim, int end_dim,
                                 torch::Tensor total_eikonal_loss,
                                 torch::Tensor total_viscosity_loss,
                                 torch::Tensor valid_loss_size,
                                 torch::Tensor dx_out,
                                 torch::Tensor dy_out,
                                 torch::Tensor dz_out,
                                 torch::Tensor dxyz_out)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(!node_gho_neighbors.is_floating_point());
    TORCH_CHECK(node_gho_coeff.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);
    TORCH_CHECK(node_gho_neighbors.ndimension() == 3);
    TORCH_CHECK(node_gho_coeff.ndimension() == 3);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::viscosity_loss_fused_kernel_LOT_T1<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_gho_neighbors.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        node_gho_coeff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scale / nl,
        epsilon,
        start_dim,
        end_dim,
        Q,
        valid_threshold,
        total_eikonal_loss.data_ptr<float>(),
        total_viscosity_loss.data_ptr<float>(),
        valid_loss_size.data_ptr<float>(),
        dx_out.data_ptr<float>(),
        dy_out.data_ptr<float>(),
        dz_out.data_ptr<float>(),
        dxyz_out.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void viscosity_loss_fused_LOT_S(torch::Tensor node_all_neighbors,
                                torch::Tensor node_all_neighlen,
                                torch::Tensor sdf_data,
                                torch::Tensor corner_indices,
                                float scale,
                                float epsilon,
                                float valid_threshold,
                                int start_dim, int end_dim,
                                torch::Tensor total_eikonal_loss,
                                torch::Tensor total_viscosity_loss,
                                torch::Tensor valid_loss_size,
                                torch::Tensor grad_data)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_all_neighlen);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(grad_data.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(node_all_neighlen.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(grad_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_all_neighlen.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::viscosity_loss_fused_kernel_LOT_S<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_all_neighlen.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scale / nl,
        epsilon,
        start_dim,
        end_dim,
        Q,
        valid_threshold,
        total_eikonal_loss.data_ptr<float>(),
        total_viscosity_loss.data_ptr<float>(),
        valid_loss_size.data_ptr<float>(),
        grad_data.data_ptr<float>());

    CUDA_CHECK_ERRORS;
}

void thin_plate_grad_LOT(torch::Tensor node_all_neighbors,
                         torch::Tensor node_dc_neighbors,
                         torch::Tensor sdf_data,
                         torch::Tensor corner_indices,
                         float scaling,
                         int start_dim, int end_dim,
                         torch::Tensor sdf_grad,
                         torch::Tensor total)
{
    DEVICE_GUARD(sdf_data);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(node_all_neighbors);
    CHECK_INPUT(node_dc_neighbors);
    CHECK_INPUT(sdf_data);
    CHECK_INPUT(corner_indices);

    TORCH_CHECK(sdf_data.is_floating_point());
    TORCH_CHECK(sdf_grad.is_floating_point());
    TORCH_CHECK(!node_all_neighbors.is_floating_point());
    TORCH_CHECK(!node_dc_neighbors.is_floating_point());
    TORCH_CHECK(sdf_data.ndimension() == 2);
    TORCH_CHECK(node_all_neighbors.ndimension() == 2);
    TORCH_CHECK(node_dc_neighbors.ndimension() == 2);
    TORCH_CHECK(sdf_grad.ndimension() == 2);

    int nl = corner_indices.size(0);
    size_t Q = corner_indices.size(0) * size_t(end_dim - start_dim);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::thin_plate_kernel_LOT<<<blocks, cuda_n_threads>>>(
        node_all_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        node_dc_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        sdf_data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        corner_indices.data_ptr<int32_t>(),
        scaling,
        start_dim,
        end_dim,
        Q,
        sdf_grad.data_ptr<float>(),
        total.data_ptr<float>());
    CUDA_CHECK_ERRORS;
}

void msi_tv_grad_sparse(
    // (reso * 2, reso)
    torch::Tensor links,
    // (capacity, n_layers, n_channels)
    torch::Tensor msi,
    torch::Tensor rand_cells,
    torch::Tensor mask_out,
    float scale,
    float scale_last,
    torch::Tensor grad_msi)
{
    DEVICE_GUARD(msi);
    CHECK_INPUT(links);
    CHECK_INPUT(msi);
    CHECK_INPUT(grad_msi);
    CHECK_INPUT(rand_cells);
    CHECK_INPUT(mask_out);
    TORCH_CHECK(msi.is_floating_point());
    TORCH_CHECK(grad_msi.is_floating_point());

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * msi.size(2);

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::msi_tv_grad_sparse_kernel<<<blocks, cuda_n_threads>>>(
        links.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        msi.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        rand_cells.data_ptr<int32_t>(),
        scale / nl,
        scale_last / nl,
        Q,
        // Output
        mask_out.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
        grad_msi.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
    CUDA_CHECK_ERRORS;
}

void lumisphere_tv_grad_sparse(
    SparseGridSpec &grid,
    torch::Tensor rand_cells,
    torch::Tensor basis_fn,
    torch::Tensor basis_fn_u,
    float scale,
    float ndc_coeffx,
    float ndc_coeffy,
    float dir_factor,
    GridOutputGrads &grads)
{
    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rand_cells);
    CHECK_INPUT(basis_fn);
    CHECK_INPUT(basis_fn_u);
    TORCH_CHECK(basis_fn.ndimension() == 1);
    grid.check();
    grads.check();

    int nl = rand_cells.size(0);
    size_t Q = rand_cells.size(0) * WARP_SIZE;

    const int cuda_n_threads = TV_GRAD_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::lumisphere_tv_grad_sparse_kernel<<<blocks, cuda_n_threads>>>(
        grid,
        rand_cells.data_ptr<int32_t>(),
        basis_fn.data_ptr<float>(),
        basis_fn_u.data_ptr<float>(),
        scale / nl,
        Q,
        ndc_coeffx, ndc_coeffy,
        dir_factor,
        // Output
        grads);
    CUDA_CHECK_ERRORS;
}
