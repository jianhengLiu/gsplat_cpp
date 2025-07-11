#include "fully_fused_projection.h"

#include <gsplat/cuda/include/Ops.h>

using namespace torch;
using namespace std;

torch::autograd::tensor_list FullyFusedProjectionPacked::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4] or None
    const torch::Tensor &scales,   // [N, 3] or None
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    int width, int height, float eps2d, float near_plane, float far_plane,
    float radius_clip, bool sparse_grad, bool calc_compensations) {
  auto camera_model_type = gsplat::CameraModelType::PINHOLE;

  auto [indptr, camera_ids, gaussian_ids, radii, means2d, depths, conics,
        compensations] =
      gsplat::projection_ewa_3dgs_packed_fwd(
          means,
          at::nullopt, // optional
          quats,       // optional
          scales,      // optional
          at::nullopt, // optional
          viewmats, Ks, width, height, eps2d, near_plane, far_plane,
          radius_clip, calc_compensations, camera_model_type);

  ctx->save_for_backward(
      {camera_ids, gaussian_ids, means, quats, scales, viewmats, Ks, conics});

  if (!calc_compensations) {
    ctx->saved_data["compensations"] = at::nullopt;
    compensations = torch::ones({means2d.size(0)}, means.options());
  } else {
    ctx->saved_data["compensations"] =
        at::optional<torch::Tensor>(compensations);
  }

  ctx->saved_data["width"] = width;
  ctx->saved_data["height"] = height;
  ctx->saved_data["eps2d"] = eps2d;
  ctx->saved_data["sparse_grad"] = sparse_grad;
  ctx->saved_data["camera_model_type"] = camera_model_type;

  return {camera_ids, gaussian_ids, radii,        means2d,
          depths,     conics,       compensations};
}

torch::autograd::tensor_list FullyFusedProjectionPacked::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs) {
  auto v_means2d = grad_outputs[3];
  auto v_depths = grad_outputs[4];
  auto v_conics = grad_outputs[5];

  auto saved = ctx->get_saved_variables();
  auto camera_ids = saved[0];
  auto gaussian_ids = saved[1];
  auto means = saved[2];
  auto quats = saved[3];
  auto scales = saved[4];
  auto viewmats = saved[5];
  auto Ks = saved[6];
  auto conics = saved[7];

  auto compensations =
      ctx->saved_data["compensations"].toOptional<torch::Tensor>();

  int width = ctx->saved_data["width"].toInt();
  int height = ctx->saved_data["height"].toInt();
  float eps2d = ctx->saved_data["eps2d"].toDouble();
  bool sparse_grad = ctx->saved_data["sparse_grad"].toBool();
  int camera_model_type = ctx->saved_data["camera_model_type"].toInt();

  auto v_compensations = at::optional<torch::Tensor>(grad_outputs[6]);
  if (compensations.has_value()) {
    v_compensations = v_compensations.value().contiguous();
  } else {
    v_compensations = at::nullopt;
  }

  auto [v_means, v_covars, v_quats, v_scales, v_viewmats] =
      gsplat::projection_ewa_3dgs_packed_bwd(
          means, at::nullopt, quats, scales, viewmats, Ks, width, height, eps2d,
          gsplat::CameraModelType(camera_model_type), camera_ids, gaussian_ids,
          conics, compensations, v_means2d.contiguous(), v_depths.contiguous(),
          v_conics.contiguous(), v_compensations,
          ctx->needs_input_grad(4), // viewmats_requires_grad
          sparse_grad);

  if (!ctx->needs_input_grad(0)) {
    v_means = torch::Tensor();
  } else if (sparse_grad) {
    // # TODO: gaussian_ids is duplicated so not ideal.
    // # An idea is to directly set the attribute (e.g., .sparse_grad) of
    // # the tensor but this requires the tensor to be leaf node only. And
    // # a customized optimizer would be needed in this case.
    v_means =
        torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_means,
                                 means.sizes(), {}, viewmats.size(0) == 1);
  }

  if (!ctx->needs_input_grad(1)) {
    v_quats = torch::Tensor();
  } else if (sparse_grad) {
    v_quats =
        torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_quats,
                                 quats.sizes(), {}, viewmats.size(0) == 1);
  }

  if (!ctx->needs_input_grad(2)) {
    v_scales = torch::Tensor();
  } else if (sparse_grad) {
    v_scales =
        torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_scales,
                                 scales.sizes(), {}, viewmats.size(0) == 1);
  }

  if (!ctx->needs_input_grad(3)) {
    v_viewmats = torch::Tensor();
  }

  return {v_means,         v_quats,         v_scales,        v_viewmats,
          torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
          torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
          torch::Tensor(), torch::Tensor()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection(torch::Tensor means,    // [N, 3]
                       torch::Tensor quats,    // [N, 4] or None
                       torch::Tensor scales,   // [N, 3] or None
                       torch::Tensor viewmats, // [C, 4, 4]
                       torch::Tensor Ks,       // [C, 3, 3]
                       int width, int height, float eps2d, float near_plane,
                       float far_plane, float radius_clip, bool packed,
                       bool sparse_grad, bool calc_compensations) {
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
      means, quats, scales, viewmats, Ks, width, height, eps2d, near_plane,
      far_plane, radius_clip, sparse_grad, calc_compensations);
  return std::make_tuple(outputs[0], outputs[1], outputs[2], outputs[3],
                         outputs[4], outputs[5], outputs[6]);
}

torch::autograd::tensor_list FullyFusedProjectionPacked2DGS::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4] or None
    const torch::Tensor &scales,   // [N, 3] or None
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    int width, int height, float near_plane, float far_plane, float radius_clip,
    bool sparse_grad) {
  auto [indptr, camera_ids, gaussian_ids, radii, means2d, depths,
        ray_transforms, normals, randns, samples] =
      gsplat::projection_2dgs_packed_fwd(means, quats, scales, viewmats, Ks,
                                         width, height, near_plane, far_plane,
                                         radius_clip);

  ctx->save_for_backward({camera_ids, gaussian_ids, means, quats, scales,
                          viewmats, Ks, ray_transforms, randns});

  ctx->saved_data["width"] = width;
  ctx->saved_data["height"] = height;
  ctx->saved_data["sparse_grad"] = sparse_grad;

  auto sample_weights = torch::exp(-0.5f * randns.square().sum(-1, true));

  return {camera_ids,     gaussian_ids, radii,   means2d,       depths,
          ray_transforms, normals,      samples, sample_weights};
}

torch::autograd::tensor_list FullyFusedProjectionPacked2DGS::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs) {
  auto v_means2d = grad_outputs[3];
  auto v_depths = grad_outputs[4];
  auto v_ray_transforms = grad_outputs[5];
  auto v_normals = grad_outputs[6];
  auto v_samples = grad_outputs[7];

  auto saved = ctx->get_saved_variables();
  auto camera_ids = saved[0];
  auto gaussian_ids = saved[1];
  auto means = saved[2];
  auto quats = saved[3];
  auto scales = saved[4];
  auto viewmats = saved[5];
  auto Ks = saved[6];
  auto ray_transforms = saved[7];
  auto randns = saved[8];

  int width = ctx->saved_data["width"].toInt();
  int height = ctx->saved_data["height"].toInt();
  bool sparse_grad = ctx->saved_data["sparse_grad"].toBool();
  auto [v_means, v_quats, v_scales, v_viewmats] =
      gsplat::projection_2dgs_packed_bwd(
          means, quats, scales, viewmats, Ks, width, height, camera_ids,
          gaussian_ids, ray_transforms, randns, v_means2d.contiguous(),
          v_depths.contiguous(), v_ray_transforms.contiguous(),
          v_normals.contiguous(), v_samples.contiguous(),
          ctx->needs_input_grad(3), // viewmats_requires_grad
          sparse_grad);

  if (!ctx->needs_input_grad(0)) {
    v_means = torch::Tensor();
  } else if (sparse_grad) {
    // # TODO: gaussian_ids is duplicated so not ideal.
    // # An idea is to directly set the attribute (e.g., .sparse_grad) of
    // # the tensor but this requires the tensor to be leaf node only. And
    // # a customized optimizer would be needed in this case.
    v_means =
        torch::sparse_coo_tensor(gaussian_ids.unsqueeze(0), v_means,
                                 means.sizes(), {}, viewmats.size(0) == 1);
  }

  if (!ctx->needs_input_grad(1)) {
    v_quats = torch::Tensor();
  } else if (sparse_grad) {
    v_quats =
        torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_quats,
                                 quats.sizes(), {}, viewmats.size(0) == 1);
  }

  if (!ctx->needs_input_grad(2)) {
    v_scales = torch::Tensor();
  } else if (sparse_grad) {
    v_scales =
        torch::sparse_coo_tensor({gaussian_ids.unsqueeze(0)}, v_scales,
                                 scales.sizes(), {}, viewmats.size(0) == 1);
  }

  if (!ctx->needs_input_grad(3)) {
    v_viewmats = torch::Tensor();
  }

  return {v_means,         v_quats,         v_scales,        v_viewmats,
          torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
          torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
          torch::Tensor(), torch::Tensor()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
fully_fused_projection_2dgs(torch::Tensor means,    // [N, 3]
                            torch::Tensor quats,    // [N, 4] or None
                            torch::Tensor scales,   // [N, 3] or None
                            torch::Tensor viewmats, // [C, 4, 4]
                            torch::Tensor Ks,       // [C, 3, 3]
                            int width, int height, float near_plane,
                            float far_plane, float radius_clip, bool packed,
                            bool sparse_grad) {
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
              "Invalid scales size: ", scales.sizes(),
              torch::IntArrayRef({N, 3}));
  quats = quats.contiguous();
  scales = scales.contiguous();

  if (sparse_grad) {
    TORCH_CHECK(packed, "sparse_grad is only supported when packed is True");
  }

  viewmats = viewmats.contiguous();
  Ks = Ks.contiguous();

  auto outputs = FullyFusedProjectionPacked2DGS::apply(
      means, quats, scales, viewmats, Ks, width, height, near_plane, far_plane,
      radius_clip, sparse_grad);
  return std::make_tuple(outputs[0], outputs[1], outputs[2], outputs[3],
                         outputs[4], outputs[5], outputs[6], outputs[7],
                         outputs[8]);
}