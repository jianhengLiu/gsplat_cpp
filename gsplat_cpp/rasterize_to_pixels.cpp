#include "rasterize_to_pixels.h"

#include <gsplat/cuda/csrc/bindings.h>

using namespace torch;
using namespace std;

torch::autograd::tensor_list RasterizeToPixels::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, D]
    const torch::Tensor &opacities,                 // [C, N]
    const at::optional<torch::Tensor> &backgrounds, // [C, D], Optional
    const at::optional<torch::Tensor>
        &masks, // [C, tile_height, tile_width], Optional
    int width, int height, int tile_size,
    const torch::Tensor &isect_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    const at::optional<torch::Tensor> &absgrad) {
  auto [render_colors, render_alphas, last_ids] =
      gsplat::rasterize_to_pixels_fwd_tensor(
          means2d, conics, colors, opacities, backgrounds, masks, width, height,
          tile_size, isect_offsets, flatten_ids);

  ctx->save_for_backward({means2d, conics, colors, opacities, isect_offsets,
                          flatten_ids, render_alphas, last_ids});
  ctx->saved_data["backgrounds"] = backgrounds;
  ctx->saved_data["masks"] = masks;
  ctx->saved_data["width"] = width;
  ctx->saved_data["height"] = height;
  ctx->saved_data["tile_size"] = tile_size;
  ctx->saved_data["absgrad"] = absgrad;

  // double to float
  render_alphas = render_alphas.to(torch::kFloat);
  return {render_colors, render_alphas};
}

torch::autograd::tensor_list
RasterizeToPixels::backward(torch::autograd::AutogradContext *ctx,
                             torch::autograd::tensor_list grad_outputs) {
  auto v_render_colors = grad_outputs[0];
  auto v_render_alphas = grad_outputs[1];

  auto saved = ctx->get_saved_variables();
  auto means2d = saved[0];
  auto conics = saved[1];
  auto colors = saved[2];
  auto opacities = saved[3];
  auto isect_offsets = saved[4];
  auto flatten_ids = saved[5];
  auto render_alphas = saved[6];
  auto last_ids = saved[7];

  auto backgrounds = ctx->saved_data["backgrounds"].toOptional<torch::Tensor>();
  auto masks = ctx->saved_data["masks"].toOptional<torch::Tensor>();

  int width = ctx->saved_data["width"].toInt();
  int height = ctx->saved_data["height"].toInt();
  int tile_size = ctx->saved_data["tile_size"].toInt();
  auto absgrad = ctx->saved_data["absgrad"].toOptional<torch::Tensor>();
  std::cout << absgrad.has_value() << std::endl;

  auto [v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities] =
      gsplat::rasterize_to_pixels_bwd_tensor(
          means2d, conics, colors, opacities, backgrounds, masks, width, height,
          tile_size, isect_offsets, flatten_ids, render_alphas, last_ids,
          v_render_colors.contiguous(), v_render_alphas.contiguous(),
          absgrad.has_value());

  std::cout << "v_means2d_abs.sizes(): " << v_means2d_abs.sizes() << std::endl;
  std::cout << absgrad.has_value() << std::endl;
  if (absgrad.has_value()) {
    absgrad = v_means2d_abs.clone();
  }

  torch::Tensor v_backgrounds = torch::Tensor();
  if (backgrounds.has_value()) {
    if (backgrounds.value().requires_grad()) {
      v_backgrounds =
          (v_render_colors * (1.0 - render_alphas).to(torch::kFloat))
              .sum({1, 2});
    }
  }

  return {
      v_means2d,       v_conics, v_colors, v_opacities, v_backgrounds,
      torch::Tensor(), // masks
      torch::Tensor(), // width
      torch::Tensor(), // height
      torch::Tensor(), // tile_size
      torch::Tensor(), // isect_offsets
      torch::Tensor(), // flatten_ids
      torch::Tensor()  // absgrad
  };
}

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_pixels(
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    torch::Tensor &colors,          // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N] or [nnz]
    int image_width, int image_height, int tile_size,
    const torch::Tensor &isect_offsets,      // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,        // [n_isects]
    at::optional<torch::Tensor> backgrounds, // [C, channels]
    at::optional<torch::Tensor> masks,       // [C, tile_height, tile_width]
    bool packed, const at::optional<torch::Tensor> &absgrad) {
  int C = isect_offsets.size(0);
  auto device = means2d.device();

  int64_t nnz = means2d.size(0);
  TORCH_CHECK(means2d.sizes() == torch::IntArrayRef({nnz, 2}),
              "Invalid shape for means2d");
  TORCH_CHECK(conics.sizes() == torch::IntArrayRef({nnz, 3}),
              "Invalid shape for conics");
  TORCH_CHECK(colors.size(0) == nnz, "Invalid shape for colors");
  TORCH_CHECK(opacities.sizes() == torch::IntArrayRef({nnz}),
              "Invalid shape for opacities");

  if (backgrounds.has_value()) {
    TORCH_CHECK(backgrounds.value().sizes() ==
                    torch::IntArrayRef({C, colors.size(-1)}),
                "Invalid shape for backgrounds");
  }
  if (masks.has_value()) {
    TORCH_CHECK(masks.value().sizes() == isect_offsets.sizes(),
                "Invalid shape for masks");
  }

  // Pad the channels to the nearest supported number if necessary
  int channels = colors.size(-1);
  if (channels > 513 || channels == 0) {
    throw std::invalid_argument("Unsupported number of color channels: " +
                                std::to_string(channels));
  }
  static std::vector<int> supported_channels = {1,   2,   3,   4,   5,  8,  9,
                                                16,  17,  32,  33,  64, 65, 128,
                                                129, 256, 257, 512, 513};
  int padded_channels = 0;
  if (std::find(supported_channels.begin(), supported_channels.end(),
                channels) == supported_channels.end()) {
    padded_channels = (1 << (int(std::log2(channels)) + 1)) - channels;
    colors = torch::cat(
        {colors, torch::zeros({colors.size(0), colors.size(1), padded_channels},
                              device)});
    if (backgrounds.has_value()) {
      backgrounds = torch::cat(
          {backgrounds.value(),
           torch::zeros({backgrounds.value().size(0), padded_channels},
                        device)});
    }
  }

  int tile_height = isect_offsets.size(1);
  int tile_width = isect_offsets.size(2);
  TORCH_CHECK(tile_height * tile_size >= image_height,
              "Assert Failed: tile_height * tile_size >= image_height");
  TORCH_CHECK(tile_width * tile_size >= image_width,
              "Assert Failed: tile_width * tile_size >= image_width");

  auto outputs = RasterizeToPixels::apply(
      means2d.contiguous(), conics.contiguous(), colors.contiguous(),
      opacities.contiguous(), backgrounds, masks, image_width, image_height,
      tile_size, isect_offsets.contiguous(), flatten_ids.contiguous(), absgrad);
  auto render_colors = outputs[0];
  auto render_alphas = outputs[1];

  if (padded_channels > 0) {
    render_colors =
        render_colors.slice(/*dim=*/-1, /*start=*/0, /*end=*/-padded_channels);
  }

  return std::make_tuple(render_colors, render_alphas);
}