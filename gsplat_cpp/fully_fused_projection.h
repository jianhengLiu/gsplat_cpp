#pragma once

#include <torch/torch.h>

struct FullyFusedProjectionPacked
    : public torch::autograd::Function<FullyFusedProjectionPacked> {
public:
  static torch::autograd::tensor_list
  forward(torch::autograd::AutogradContext *ctx,
          const torch::Tensor &means,    // [N, 3]
          const torch::Tensor &quats,    // [N, 4] or None
          const torch::Tensor &scales,   // [N, 3] or None
          const torch::Tensor &viewmats, // [C, 4, 4]
          const torch::Tensor &Ks,       // [C, 3, 3]
          int width, int height, float eps2d, float near_plane, float far_plane,
          float radius_clip, bool sparse_grad, bool calc_compensations);

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs);
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
fully_fused_projection(torch::Tensor means,    // [N, 3]
                       torch::Tensor quats,    // [N, 4] or None
                       torch::Tensor scales,   // [N, 3] or None
                       torch::Tensor viewmats, // [C, 4, 4]
                       torch::Tensor Ks,       // [C, 3, 3]
                       int width, int height, float eps2d = 0.3f,
                       float near_plane = 0.01f, float far_plane = 1e10f,
                       float radius_clip = 0.0f, bool packed = false,
                       bool sparse_grad = false,
                       bool calc_compensations = false);

struct FullyFusedProjectionPacked2DGS
    : public torch::autograd::Function<FullyFusedProjectionPacked2DGS> {
public:
  static torch::autograd::tensor_list
  forward(torch::autograd::AutogradContext *ctx,
          const torch::Tensor &means,    // [N, 3]
          const torch::Tensor &quats,    // [N, 4] or None
          const torch::Tensor &scales,   // [N, 3] or None
          const torch::Tensor &viewmats, // [C, 4, 4]
          const torch::Tensor &Ks,       // [C, 3, 3]
          int width, int height, float near_plane, float far_plane,
          float radius_clip, bool sparse_grad);

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs);
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
fully_fused_projection_2dgs(torch::Tensor means,    // [N, 3]
                            torch::Tensor quats,    // [N, 4] or None
                            torch::Tensor scales,   // [N, 3] or None
                            torch::Tensor viewmats, // [C, 4, 4]
                            torch::Tensor Ks,       // [C, 3, 3]
                            int width, int height, float near_plane = 0.01f,
                            float far_plane = 1e10f, float radius_clip = 0.0f,
                            bool packed = false, bool sparse_grad = false);