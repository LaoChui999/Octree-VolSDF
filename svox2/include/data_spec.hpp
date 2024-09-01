// Copyright 2021 Alex Yu
#pragma once
#include "util.hpp"
#include <torch/extension.h>

using torch::Tensor;

enum BasisType
{
  // For svox 1 compatibility
  // BASIS_TYPE_RGBA = 0
  BASIS_TYPE_SH = 1,
  // BASIS_TYPE_SG = 2
  // BASIS_TYPE_ASG = 3
  BASIS_TYPE_3D_TEXTURE = 4,
  BASIS_TYPE_MLP = 255,
};

struct SparseGridSpec
{
  Tensor density_data;
  Tensor sh_data;
  Tensor links;
  Tensor _offset;
  Tensor _scaling;

  Tensor background_links;
  Tensor background_data;

  int basis_dim;
  uint8_t basis_type;
  Tensor basis_data;

  inline void check()
  {
    CHECK_INPUT(density_data);
    CHECK_INPUT(sh_data);
    CHECK_INPUT(links);
    if (background_links.defined())
    {
      CHECK_INPUT(background_links);
      CHECK_INPUT(background_data);
      TORCH_CHECK(background_links.ndimension() ==
                  2);                                 // (H, W) -> [N] \cup {-1}
      TORCH_CHECK(background_data.ndimension() == 3); // (N, D, C) -> R
    }
    if (basis_data.defined())
    {
      CHECK_INPUT(basis_data);
    }
    CHECK_CPU_INPUT(_offset);
    CHECK_CPU_INPUT(_scaling);
    TORCH_CHECK(density_data.ndimension() == 2);
    TORCH_CHECK(sh_data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);
  }
};

struct TreeSpecLOT
{
  Tensor child;
  Tensor _offset;
  Tensor _scaling;
  Tensor CornerSH;
  Tensor CornerD;
  Tensor CornerSDF;
  Tensor CornerGaussSDF;
  Tensor data;
  Tensor LearnS;
  Tensor Beta;
  Tensor NodeCorners;
  Tensor NodeAllNeighbors;
  Tensor NodeAllNeighLen;

  int basis_dim;
  uint8_t basis_type;

  inline void check()
  {
    CHECK_INPUT(child);
    CHECK_INPUT(CornerSH);
    CHECK_INPUT(CornerD);
    CHECK_INPUT(CornerSDF);
    CHECK_INPUT(CornerGaussSDF);
    CHECK_INPUT(LearnS);
    CHECK_INPUT(Beta);
    CHECK_INPUT(NodeCorners);
    CHECK_INPUT(NodeAllNeighbors);
    CHECK_INPUT(NodeAllNeighLen);

    CHECK_CPU_INPUT(_offset);
    CHECK_CPU_INPUT(_scaling);

    TORCH_CHECK(child.ndimension() == 4);
    //TORCH_CHECK(CornerSH.ndimension() == 2);
    TORCH_CHECK(CornerD.ndimension() == 2);
    //TORCH_CHECK(CornerSDF.ndimension() == 2);
    TORCH_CHECK(CornerGaussSDF.ndimension() == 2);
    TORCH_CHECK(NodeCorners.ndimension() == 5);
    TORCH_CHECK(NodeAllNeighbors.ndimension() == 2);
    TORCH_CHECK(NodeAllNeighLen.ndimension() == 2);
  }
};

struct GridOutputGrads
{
  torch::Tensor grad_density_out;
  torch::Tensor grad_sh_out;
  torch::Tensor grad_basis_out;
  torch::Tensor grad_background_out;

  torch::Tensor mask_out;
  torch::Tensor mask_background_out;
  inline void check()
  {
    if (grad_density_out.defined())
    {
      CHECK_INPUT(grad_density_out);
    }
    if (grad_sh_out.defined())
    {
      CHECK_INPUT(grad_sh_out);
    }
    if (grad_basis_out.defined())
    {
      CHECK_INPUT(grad_basis_out);
    }
    if (grad_background_out.defined())
    {
      CHECK_INPUT(grad_background_out);
    }
    if (mask_out.defined() && mask_out.size(0) > 0)
    {
      CHECK_INPUT(mask_out);
    }
    if (mask_background_out.defined() && mask_background_out.size(0) > 0)
    {
      CHECK_INPUT(mask_background_out);
    }
  }
};

struct GridOutputGradsLOT
{
  torch::Tensor grad_density_out;
  torch::Tensor grad_sh_out;

  inline void check()
  {
    if (grad_density_out.defined())
    {
      CHECK_INPUT(grad_density_out);
    }
    if (grad_sh_out.defined())
    {
      CHECK_INPUT(grad_sh_out);
    }
  }
};

struct GridOutputGradsSDFLOT
{
  torch::Tensor grad_sdf_out;
  torch::Tensor grad_learns_out;
  torch::Tensor grad_beta_out;
  torch::Tensor grad_sh_out;

  inline void check()
  {
    if (grad_sdf_out.defined())
    {
      CHECK_INPUT(grad_sdf_out);
    }

    if (grad_learns_out.defined())
    {
      CHECK_INPUT(grad_learns_out);
    }

    if (grad_beta_out.defined())
    {
      CHECK_INPUT(grad_beta_out);
    }


    if (grad_sh_out.defined())
    {
      CHECK_INPUT(grad_sh_out);
    }
  }
};

struct CameraSpec
{
  torch::Tensor c2w;
  float fx;
  float fy;
  float cx;
  float cy;
  int width;
  int height;

  float ndc_coeffx;
  float ndc_coeffy;

  inline void check()
  {
    CHECK_INPUT(c2w);
    TORCH_CHECK(c2w.is_floating_point());
    TORCH_CHECK(c2w.ndimension() == 2);
    TORCH_CHECK(c2w.size(1) == 4);
  }
};

struct RaysSpec
{
  Tensor origins;
  Tensor dirs;
  inline void check()
  {
    CHECK_INPUT(origins);
    CHECK_INPUT(dirs);
    TORCH_CHECK(origins.is_floating_point());
    TORCH_CHECK(dirs.is_floating_point());
  }
};

struct RaysHitLOTreeSDF
{
  Tensor sdf_point;
  Tensor col_point;
  Tensor hitnode_sdf;
  Tensor hitnode_col;
  Tensor hitnum;

  inline void check()
  {
    CHECK_INPUT(sdf_point);
    CHECK_INPUT(col_point);
    CHECK_INPUT(hitnode_sdf);
    CHECK_INPUT(hitnode_col);
    CHECK_INPUT(hitnum);
  }
};

struct RenderOptions
{
  float background_brightness;
  // float step_epsilon;
  float step_size;
  float t_step;
  float sigma_thresh;
  float sdf_thresh;
  float cube_thresh;
  float alpha_thresh;
  float stop_thresh;

  float near_clip;
  bool use_spheric_clip;

  bool last_sample_opaque;

  float step_sigma;
  float step_K;
  float step_b;

  int sample_size;

  // bool randomize;
  // float random_sigma_std;
  // float random_sigma_std_background;
  // 32-bit RNG state masks
  // uint32_t _m1, _m2, _m3;

  // int msi_start_layer = 0;
  // int msi_end_layer = 66;
};
