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
    const torch::Tensor &absgrad) {
  auto [render_colors, render_alphas, last_ids] =
      gsplat::rasterize_to_pixels_fwd_tensor(
          means2d, conics, colors, opacities, backgrounds, masks, width, height,
          tile_size, isect_offsets, flatten_ids);

  ctx->save_for_backward({means2d, conics, colors, opacities, isect_offsets,
                          flatten_ids, render_alphas, last_ids, absgrad});
  ctx->saved_data["backgrounds"] = backgrounds;
  ctx->saved_data["masks"] = masks;
  ctx->saved_data["width"] = width;
  ctx->saved_data["height"] = height;
  ctx->saved_data["tile_size"] = tile_size;

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
  auto absgrad = saved[8];

  auto backgrounds = ctx->saved_data["backgrounds"].toOptional<torch::Tensor>();
  auto masks = ctx->saved_data["masks"].toOptional<torch::Tensor>();

  int width = ctx->saved_data["width"].toInt();
  int height = ctx->saved_data["height"].toInt();
  int tile_size = ctx->saved_data["tile_size"].toInt();

  auto [v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities] =
      gsplat::rasterize_to_pixels_bwd_tensor(
          means2d, conics, colors, opacities, backgrounds, masks, width, height,
          tile_size, isect_offsets, flatten_ids, render_alphas, last_ids,
          v_render_colors.contiguous(), v_render_alphas.contiguous(),
          absgrad.defined());

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
      v_means2d_abs    // absgrad
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
    bool packed, const torch::Tensor &absgrad) {
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
    auto color_sizes = colors.sizes().slice(0, -1).vec();
    color_sizes.emplace_back(padded_channels);
    colors = torch::cat({colors, torch::zeros(color_sizes, device)}, -1);
    if (backgrounds.has_value()) {
      auto backgrounds_sizes = backgrounds.value().sizes().slice(0, -1).vec();
      backgrounds_sizes.emplace_back(padded_channels);
      backgrounds = torch::cat(
          {backgrounds.value(), torch::zeros(backgrounds_sizes, device)});
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

torch::autograd::tensor_list RasterizeToPixels2DGS::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &ray_transforms,            // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, D]
    const torch::Tensor &opacities,                 // [C, N]
    const torch::Tensor &normals,                   // [C, N]
    const torch::Tensor &densify,                   // [C, N]
    const at::optional<torch::Tensor> &backgrounds, // [C, D], Optional
    const at::optional<torch::Tensor>
        &masks, // [C, tile_height, tile_width], Optional
    int width, int height, int tile_size,
    const torch::Tensor &isect_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    const torch::Tensor &absgrad, const bool &distloss) {
  auto [render_colors, render_alphas, render_normals, render_distort,
        render_median, last_ids, median_ids] =
      gsplat::rasterize_to_pixels_fwd_2dgs_tensor(
          means2d, ray_transforms, colors, opacities, normals, backgrounds,
          masks, width, height, tile_size, isect_offsets, flatten_ids);

  ctx->save_for_backward({means2d, ray_transforms, colors, opacities, normals,
                          densify, isect_offsets, flatten_ids, render_colors,
                          render_alphas, last_ids, median_ids, absgrad});
  ctx->saved_data["backgrounds"] = backgrounds;
  ctx->saved_data["masks"] = masks;
  ctx->saved_data["width"] = width;
  ctx->saved_data["height"] = height;
  ctx->saved_data["tile_size"] = tile_size;

  // double to float
  render_alphas = render_alphas.to(torch::kFloat);
  return {render_colors, render_alphas.to(torch::kFloat), render_normals,
          render_distort, render_median};
}

torch::autograd::tensor_list
RasterizeToPixels2DGS::backward(torch::autograd::AutogradContext *ctx,
                                torch::autograd::tensor_list grad_outputs) {
  auto v_render_colors = grad_outputs[0];
  auto v_render_alphas = grad_outputs[1];
  auto v_render_normals = grad_outputs[2];
  auto v_render_distort = grad_outputs[3];
  auto v_render_median = grad_outputs[4];

  auto saved = ctx->get_saved_variables();
  auto means2d = saved[0];
  auto ray_transforms = saved[1];
  auto colors = saved[2];
  auto opacities = saved[3];
  auto normals = saved[4];
  auto densify = saved[5];
  auto isect_offsets = saved[6];
  auto flatten_ids = saved[7];
  auto render_colors = saved[8];
  auto render_alphas = saved[9];
  auto last_ids = saved[10];
  auto median_ids = saved[11];
  auto absgrad = saved[12];

  auto backgrounds = ctx->saved_data["backgrounds"].toOptional<torch::Tensor>();
  auto masks = ctx->saved_data["masks"].toOptional<torch::Tensor>();

  int width = ctx->saved_data["width"].toInt();
  int height = ctx->saved_data["height"].toInt();
  int tile_size = ctx->saved_data["tile_size"].toInt();

  auto [v_means2d_abs, v_means2d, v_ray_transforms, v_colors, v_opacities,
        v_normals, v_densify] =
      gsplat::rasterize_to_pixels_bwd_2dgs_tensor(
          means2d, ray_transforms, colors, opacities, normals, densify,
          backgrounds, masks, width, height, tile_size, isect_offsets,
          flatten_ids, render_colors, render_alphas, last_ids, median_ids,
          v_render_colors.contiguous(), v_render_alphas.contiguous(),
          v_render_normals.contiguous(), v_render_distort.contiguous(),
          v_render_median.contiguous(), absgrad.requires_grad());
  torch::cuda::synchronize();

  torch::Tensor v_backgrounds = torch::Tensor();
  if (backgrounds.has_value()) {
    if (backgrounds.value().requires_grad()) {
      v_backgrounds =
          (v_render_colors * (1.0 - render_alphas).to(torch::kFloat))
              .sum({1, 2});
    }
  }

  return {
      v_means2d,       v_ray_transforms, v_colors,      v_opacities,
      v_normals,       v_densify,        v_backgrounds,
      torch::Tensor(), // masks
      torch::Tensor(), // width
      torch::Tensor(), // height
      torch::Tensor(), // tile_size
      torch::Tensor(), // isect_offsets
      torch::Tensor(), // flatten_ids
      v_means2d_abs,   // absgrad
      torch::Tensor(), // distloss
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
rasterize_to_pixels_2dgs(
    const torch::Tensor &means2d,        // [C, N, 2] or [nnz, 2]
    const torch::Tensor &ray_transforms, // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,         // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,      // [C, N] or [nnz]
    const torch::Tensor &normals,        // [C, N, 3] or [nnz, 3]
    const torch::Tensor &densify,        // [C, N, 2] or [nnz, 2]
    int image_width, int image_height, int tile_size,
    const torch::Tensor &isect_offsets,      // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,        // [n_isects]
    at::optional<torch::Tensor> backgrounds, // [C, channels]
    at::optional<torch::Tensor> masks,       // [C, tile_height, tile_width]
    bool packed, const torch::Tensor &absgrad, bool distloss) {
  int C = isect_offsets.size(0);
  auto device = means2d.device();

  int64_t nnz = means2d.size(0);
  TORCH_CHECK(means2d.sizes() == torch::IntArrayRef({nnz, 2}),
              "Invalid shape for means2d");
  TORCH_CHECK(ray_transforms.sizes() == torch::IntArrayRef({nnz, 3, 3}),
              "Invalid shape for conics");
  TORCH_CHECK(colors.size(0) == nnz, "Invalid shape for colors", colors.size(0),
              ", ", nnz);
  TORCH_CHECK(opacities.sizes() == torch::IntArrayRef({nnz}),
              "Invalid shape for opacities", opacities.sizes(), ", ", nnz);

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
  if (channels > 512 || channels == 0) {
    throw std::invalid_argument("Unsupported number of color channels: " +
                                std::to_string(channels));
  }
  static std::vector<int> supported_channels = {1,  2,  3,   4,   8,  16,
                                                32, 64, 128, 256, 512};
  int padded_channels = 0;
  auto pad_colors = colors;
  if (std::find(supported_channels.begin(), supported_channels.end(),
                channels) == supported_channels.end()) {
    padded_channels = (1 << (int(std::log2(channels)) + 1)) - channels;
    auto color_sizes = colors.sizes().slice(0, -1).vec();
    color_sizes.emplace_back(padded_channels);
    pad_colors = torch::cat({colors, torch::zeros(color_sizes, device)}, -1);
    if (backgrounds.has_value()) {
      auto backgrounds_sizes = backgrounds.value().sizes().slice(0, -1).vec();
      backgrounds_sizes.emplace_back(padded_channels);
      backgrounds = torch::cat(
          {backgrounds.value(), torch::zeros(backgrounds_sizes, device)});
    }
  }

  int tile_height = isect_offsets.size(1);
  int tile_width = isect_offsets.size(2);
  TORCH_CHECK(tile_height * tile_size >= image_height,
              "Assert Failed: tile_height * tile_size >= image_height");
  TORCH_CHECK(tile_width * tile_size >= image_width,
              "Assert Failed: tile_width * tile_size >= image_width");

  auto outputs = RasterizeToPixels2DGS::apply(
      means2d.contiguous(), ray_transforms.contiguous(),
      pad_colors.contiguous(), opacities.contiguous(), normals.contiguous(),
      densify.contiguous(), backgrounds, masks, image_width, image_height,
      tile_size, isect_offsets.contiguous(), flatten_ids.contiguous(), absgrad,
      distloss);
  auto render_colors = outputs[0];
  auto render_alphas = outputs[1];
  auto render_normals = outputs[2];
  auto render_distort = outputs[3];
  auto render_median = outputs[4];

  if (padded_channels > 0) {
    render_colors =
        render_colors.slice(/*dim=*/-1, /*start=*/0, /*end=*/-padded_channels);
  }

  return std::make_tuple(render_colors, render_alphas, render_normals,
                         render_distort, render_median);
}