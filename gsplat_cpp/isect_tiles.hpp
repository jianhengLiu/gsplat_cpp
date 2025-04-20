#pragma once

#include <ATen/core/grad_mode.h>
#include <gsplat/cuda/include/Ops.h>

using namespace std;

std::tuple<at::Tensor, at::Tensor, at::Tensor>
isect_tiles(const at::Tensor &means2d, // [C, N, 2] or [nnz, 2]
            const at::Tensor &radii,   // [C, N] or [nnz]
            const at::Tensor &depths,  // [C, N] or [nnz]
            uint32_t tile_size, uint32_t tile_width, uint32_t tile_height,
            bool sort = true, bool packed = false, int n_cameras = -1,
            const at::Tensor &camera_ids = at::Tensor(),
            const at::Tensor &gaussian_ids = at::Tensor()) {
  at::NoGradGuard no_grad;

  uint32_t C;
  int64_t nnz = means2d.size(0);
  TORCH_CHECK(means2d.sizes() == at::IntArrayRef({nnz, 2}),
              "Invalid shape for means2d");
  TORCH_CHECK(radii.sizes() == at::IntArrayRef({nnz, 2}),
              "Invalid shape for radii: ", radii.sizes(),
              at::IntArrayRef({nnz}));
  TORCH_CHECK(depths.sizes() == at::IntArrayRef({nnz}),
              "Invalid shape for depths");
  TORCH_CHECK(camera_ids.defined(), "camera_ids is required if packed is True");
  TORCH_CHECK(gaussian_ids.defined(),
              "gaussian_ids is required if packed is True");
  TORCH_CHECK(n_cameras > 0, "n_cameras is required if packed is True");
  C = n_cameras;

  return gsplat::intersect_tile(means2d.contiguous(), radii.contiguous(),
                                depths.contiguous(), camera_ids.contiguous(),
                                gaussian_ids.contiguous(), C, tile_size,
                                tile_width, tile_height, sort);
}

at::Tensor isect_offset_encode(const at::Tensor &isect_ids, // [n_isects]
                               uint32_t n_cameras, uint32_t tile_width,
                               uint32_t tile_height) {
  at::NoGradGuard no_grad;
  // Call the CUDA function
  return gsplat::intersect_offset(isect_ids.contiguous(), n_cameras, tile_width,
                                  tile_height);
}