#pragma once

#include <torch/torch.h>

struct RasterizeToPixels : public torch::autograd::Function<RasterizeToPixels> {
public:
  static torch::autograd::tensor_list
  forward(torch::autograd::AutogradContext *ctx,
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
          const torch::Tensor &absgrad = torch::Tensor());

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs);
};

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_pixels(
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    torch::Tensor &colors,          // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N] or [nnz]
    int image_width, int image_height, int tile_size,
    const torch::Tensor &isect_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    at::optional<torch::Tensor> backgrounds = at::nullopt, // [C, channels]
    at::optional<torch::Tensor> masks =
        at::nullopt, // [C, tile_height, tile_width]
    bool packed = false, const torch::Tensor &absgrad = torch::Tensor());

struct RasterizeToPixels2DGS
    : public torch::autograd::Function<RasterizeToPixels2DGS> {
public:
  static torch::autograd::tensor_list
  forward(torch::autograd::AutogradContext *ctx,
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
          const torch::Tensor &absgrad = torch::Tensor(), // [C, N, 2]
          const bool &distloss = false);

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs);
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor>
rasterize_to_pixels_2dgs(
    const torch::Tensor &means2d,        // [C, N, 2] or [nnz, 2]
    const torch::Tensor &ray_transforms, // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,         // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,      // [C, N] or [nnz]
    const torch::Tensor &normals,        // [C, N, 3] or [nnz, 3]
    const torch::Tensor &densify,        // [C, N, 2] or [nnz, 2]
    int image_width, int image_height, int tile_size,
    const torch::Tensor &isect_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    at::optional<torch::Tensor> backgrounds = at::nullopt, // [C, channels]
    at::optional<torch::Tensor> masks =
        at::nullopt, // [C, tile_height, tile_width]
    bool packed = false, const torch::Tensor &absgrad = torch::Tensor(),
    bool distloss = false);