#pragma once

#include <gsplat/cuda/csrc/bindings.h>

using namespace std;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles(const torch::Tensor &means2d, // [C, N, 2] or [nnz, 2]
            const torch::Tensor &radii,   // [C, N] or [nnz]
            const torch::Tensor &depths,  // [C, N] or [nnz]
            int tile_size, int tile_width, int tile_height, bool sort = true,
            bool packed = false, int n_cameras = -1,
            const torch::Tensor &camera_ids = torch::Tensor(),
            const torch::Tensor &gaussian_ids = torch::Tensor()) {
  torch::NoGradGuard no_grad;

  int C;
  int64_t nnz = means2d.size(0);
  TORCH_CHECK(means2d.sizes() == torch::IntArrayRef({nnz, 2}),
              "Invalid shape for means2d");
  TORCH_CHECK(radii.sizes() == torch::IntArrayRef({nnz}),
              "Invalid shape for radii");
  TORCH_CHECK(depths.sizes() == torch::IntArrayRef({nnz}),
              "Invalid shape for depths");
  TORCH_CHECK(camera_ids.defined(), "camera_ids is required if packed is True");
  TORCH_CHECK(gaussian_ids.defined(),
              "gaussian_ids is required if packed is True");
  TORCH_CHECK(n_cameras > 0, "n_cameras is required if packed is True");
  C = n_cameras;

  return gsplat::isect_tiles_tensor(
      means2d.contiguous(), radii.contiguous(), depths.contiguous(),
      camera_ids.contiguous(), gaussian_ids.contiguous(), C, tile_size,
      tile_width, tile_height, sort,
      true // DoubleBuffer: memory efficient radixsort
  );
}

torch::Tensor isect_offset_encode(const torch::Tensor &isect_ids, // [n_isects]
                                  int n_cameras, int tile_width,
                                  int tile_height) {
  torch::NoGradGuard no_grad;

  // Call the CUDA function
  return gsplat::isect_offset_encode_tensor(isect_ids.contiguous(), n_cameras,
                                            tile_width, tile_height);
}