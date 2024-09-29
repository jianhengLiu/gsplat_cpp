#pragma once

#include <gsplat/cuda/csrc/bindings.h>
#include <optional>
#include <torch/torch.h>
#include <tuple>

using namespace torch;
using namespace std;

struct SphericalHarmonics
    : public torch::autograd::Function<SphericalHarmonics> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               int sh_degree, const torch::Tensor &dirs,
                               const torch::Tensor &coeffs,
                               const at::optional<torch::Tensor> &masks) {
    auto colors = gsplat::compute_sh_fwd_tensor(sh_degree, dirs, coeffs, masks);
    ctx->save_for_backward({dirs, coeffs});

    ctx->saved_data["masks"] = masks;
    ctx->saved_data["sh_degree"] = sh_degree;
    ctx->saved_data["num_bases"] = coeffs.size(-2);
    return colors;
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto dirs = saved[0];
    auto coeffs = saved[1];
    auto masks = ctx->saved_data["masks"].toOptional<torch::Tensor>();
    int sh_degree = ctx->saved_data["sh_degree"].toInt();
    int num_bases = ctx->saved_data["num_bases"].toInt();
    bool compute_v_dirs = ctx->needs_input_grad(1);

    auto v_colors = grad_outputs[0].contiguous();
    auto [v_coeffs, v_dirs] = gsplat::compute_sh_bwd_tensor(
        num_bases, sh_degree, dirs, coeffs, masks, v_colors, compute_v_dirs);

    if (!compute_v_dirs) {
      v_dirs = torch::Tensor();
    }

    return {torch::Tensor(), v_dirs, v_coeffs, torch::Tensor()};
  }
};

torch::Tensor
spherical_harmonics(int degrees_to_use,
                    torch::Tensor dirs,   // [..., 3]
                    torch::Tensor coeffs, // [..., K, 3]
                    at::optional<torch::Tensor> masks = at::nullopt) {
  // Check the input tensor shapes
  TORCH_CHECK((degrees_to_use + 1) * (degrees_to_use + 1) <= coeffs.size(-2),
              "Invalid coeffs shape");
  TORCH_CHECK(dirs.sizes().slice(0, dirs.dim() - 1) ==
                  coeffs.sizes().slice(0, coeffs.dim() - 2),
              "Shape mismatch between dirs and coeffs");
  TORCH_CHECK(dirs.size(-1) == 3,
              "dirs must have size 3 in the last dimension");
  TORCH_CHECK(coeffs.size(-1) == 3,
              "coeffs must have size 3 in the last dimension");

  // Make tensors contiguous
  dirs = dirs.contiguous();
  coeffs = coeffs.contiguous();

  if (masks.has_value()) {
    TORCH_CHECK(masks.value().sizes() == dirs.sizes().slice(0, dirs.dim() - 1),
                "Shape mismatch between masks and dirs");
    masks = masks.value().contiguous();
  }

  // Call the custom autograd function
  return SphericalHarmonics::apply(degrees_to_use, dirs, coeffs, masks);
}