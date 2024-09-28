#pragma once

#include <gsplat/cuda/csrc/bindings.h>
#include <optional>
#include <torch/torch.h>
#include <tuple>

using namespace torch;
using namespace std;

struct FullyFusedProjectionPacked
    : public torch::autograd::Function<FullyFusedProjectionPacked> {
public:
  static torch::autograd::tensor_list
  forward(torch::autograd::AutogradContext *ctx,
          torch::Tensor means,                       // [N, 3]
          const at::optional<torch::Tensor> &covars, // [N, 6] or None
          const at::optional<torch::Tensor> &quats,  // [N, 4] or None
          const at::optional<torch::Tensor> &scales, // [N, 3] or None
          torch::Tensor viewmats,                    // [C, 4, 4]
          torch::Tensor Ks,                          // [C, 3, 3]
          int width, int height, float eps2d, float near_plane, float far_plane,
          float radius_clip, bool sparse_grad, bool calc_compensations,
          const std::string &camera_model = "pinhole") {

    auto camera_model_type = gsplat::CameraModelType::PINHOLE;

    auto [indptr, camera_ids, gaussian_ids, radii, means2d, depths, conics,
          compensations] =
        gsplat::fully_fused_projection_packed_fwd_tensor(
            means,
            covars, // optional
            quats,  // optional
            scales, // optional
            viewmats, Ks, width, height, eps2d, near_plane, far_plane,
            radius_clip, calc_compensations, camera_model_type);

    if (!calc_compensations) {
      compensations = torch::Tensor();
    }

    ctx->save_for_backward(
        {camera_ids, gaussian_ids, means, viewmats, Ks, conics, compensations});

    ctx->saved_data["covars"] = covars;
    ctx->saved_data["quats"] = quats;
    ctx->saved_data["scales"] = scales;

    ctx->saved_data["width"] = width;
    ctx->saved_data["height"] = height;
    ctx->saved_data["eps2d"] = eps2d;
    ctx->saved_data["sparse_grad"] = sparse_grad;
    ctx->saved_data["camera_model_type"] = camera_model_type;

    return {camera_ids, gaussian_ids, radii,        means2d,
            depths,     conics,       compensations};
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    auto v_means2d = grad_outputs[3];
    auto v_depths = grad_outputs[4];
    auto v_conics = grad_outputs[5];
    auto v_compensations = grad_outputs[6];

    auto saved = ctx->get_saved_variables();
    auto camera_ids = saved[0];
    auto gaussian_ids = saved[1];
    auto means = saved[2];
    auto viewmats = saved[3];
    auto Ks = saved[4];
    auto conics = saved[5];
    auto compensations = saved[6];

    auto covars = ctx->saved_data["covars"].toOptional<torch::Tensor>();
    auto quats = ctx->saved_data["quats"].toOptional<torch::Tensor>();
    auto scales = ctx->saved_data["scales"].toOptional<torch::Tensor>();

    int width = ctx->saved_data["width"].toInt();
    int height = ctx->saved_data["height"].toInt();
    float eps2d = ctx->saved_data["eps2d"].toDouble();
    bool sparse_grad = ctx->saved_data["sparse_grad"].toBool();
    int camera_model_type = ctx->saved_data["camera_model_type"].toInt();

    if (v_compensations.defined()) {
      v_compensations = v_compensations.contiguous();
    }

    auto [v_means, v_covars, v_quats, v_scales, v_viewmats] =
        gsplat::fully_fused_projection_packed_bwd_tensor(
            means, covars, quats, scales, viewmats, Ks, width, height, eps2d,
            gsplat::CameraModelType(camera_model_type), camera_ids,
            gaussian_ids, conics, compensations, v_means2d.contiguous(),
            v_depths.contiguous(), v_conics.contiguous(), v_compensations,
            ctx->needs_input_grad(4), // viewmats_requires_grad
            sparse_grad);

    if (!ctx->needs_input_grad(0)) {
      v_means = torch::Tensor();
    } else if (sparse_grad) {
      // # TODO: gaussian_ids is duplicated so not ideal.
      // # An idea is to directly set the attribute (e.g., .sparse_grad) of
      // # the tensor but this requires the tensor to be leaf node only. And
      // # a customized optimizer would be needed in this case.
      v_means = torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_means,
                                         means.sizes());
    }

    if (!ctx->needs_input_grad(1)) {
      v_covars = torch::Tensor();
    } else if (sparse_grad) {
      v_covars = torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_covars,
                                          covars.value().sizes());
    }

    if (!ctx->needs_input_grad(2)) {
      v_quats = torch::Tensor();
    } else if (sparse_grad) {
      v_quats = torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_quats,
                                         quats.value().sizes());
    }

    if (!ctx->needs_input_grad(3)) {
      v_scales = torch::Tensor();
    } else if (sparse_grad) {
      v_scales = torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_scales,
                                          scales.value().sizes());
    }

    if (!ctx->needs_input_grad(4)) {
      v_viewmats = torch::Tensor();
    }

    return {v_means,         v_covars,        v_quats,         v_scales,
            v_viewmats,      torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection(Tensor means,    // [N, 3]
                       Tensor quats,    // [N, 4] or None
                       Tensor scales,   // [N, 3] or None
                       Tensor viewmats, // [C, 4, 4]
                       Tensor Ks,       // [C, 3, 3]
                       int width, int height, float eps2d = 0.3,
                       float near_plane = 0.01, float far_plane = 1e10,
                       float radius_clip = 0.0, bool packed = false,
                       bool sparse_grad = false,
                       bool calc_compensations = false) {
  int C = viewmats.size(0);
  int N = means.size(0);
  TORCH_CHECK(means.sizes() == torch::IntArrayRef({N, 3}),
              "Invalid means size");
  TORCH_CHECK(viewmats.sizes() == torch::IntArrayRef({C, 4, 4}),
              "Invalid viewmats size");
  TORCH_CHECK(Ks.sizes() == torch::IntArrayRef({C, 3, 3}), "Invalid Ks size");

  means = means.contiguous();
  TORCH_CHECK(quats.sizes() == torch::IntArrayRef({N, 4}),
              "Invalid quats size");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({N, 3}),
              "Invalid scales size");
  quats = quats.contiguous();
  scales = scales.contiguous();

  if (sparse_grad) {
    TORCH_CHECK(packed, "sparse_grad is only supported when packed is True");
  }

  viewmats = viewmats.contiguous();
  Ks = Ks.contiguous();

  auto outputs = FullyFusedProjectionPacked::apply(
      means, Tensor(), quats, scales, viewmats, Ks, width, height, eps2d,
      near_plane, far_plane, radius_clip, sparse_grad, calc_compensations);
  return std::make_tuple(outputs[0], outputs[1], outputs[2], outputs[3],
                         outputs[4], outputs[5], outputs[6]);
}