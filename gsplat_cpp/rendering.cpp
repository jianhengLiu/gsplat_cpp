#include "rendering.h"
#include <map>
#include <string>
#include <torch/torch.h>
#include <tuple>

#include "fully_fused_projection.hpp"
#include "isect_tiles.hpp"
#include "rasterize_to_pixels.h"
#include "spherical_harmonics.hpp"
namespace gsplat_cpp {
std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>>
rasterization(const torch::Tensor &means,     //[N, 3]
              const torch::Tensor &quats,     // [N, 4]
              const torch::Tensor &scales,    // [N, 3]
              const torch::Tensor &opacities, // [N]
              const torch::Tensor &colors,    //[(C,) N, D] or [(C,) N, K, 3]
              const torch::Tensor &viewmats,  //[C, 4, 4]
              const torch::Tensor &Ks,        //[C, 3, 3]
              int width, int height, const std::string &render_mode,
              float near_plane, float far_plane, float radius_clip, float eps2d,
              at::optional<int> sh_degree, bool packed, int tile_size,
              at::optional<torch::Tensor> backgrounds, bool sparse_grad,
              bool absgrad, const std::string &rasterize_mode,
              int channel_chunk) {
  std::map<std::string, torch::Tensor> meta;

  auto N = means.size(0);
  auto C = viewmats.size(0);
  auto device = means.device();

  TORCH_CHECK(means.sizes() == torch::IntArrayRef({N, 3}),
              "Invalid means shape");
  TORCH_CHECK(quats.sizes() == torch::IntArrayRef({N, 4}),
              "Invalid quats shape");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({N, 3}),
              "Invalid scales shape");
  TORCH_CHECK(opacities.sizes() == torch::IntArrayRef({N}),
              "Invalid opacities shape", opacities.sizes(),
              torch::IntArrayRef({N}));
  TORCH_CHECK(viewmats.sizes() == torch::IntArrayRef({C, 4, 4}),
              "Invalid viewmats shape");
  TORCH_CHECK(Ks.sizes() == torch::IntArrayRef({C, 3, 3}), "Invalid Ks shape");
  TORCH_CHECK(render_mode == "RGB" || render_mode == "D" ||
                  render_mode == "ED" || render_mode == "RGB+D" ||
                  render_mode == "RGB+ED",
              "Invalid render_mode");

  if (sh_degree.has_value()) {
    // # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N,
    // K, 3] # Allowing for activating partial SH bands
    TORCH_CHECK(
        (colors.dim() == 3 && colors.size(0) == N && colors.size(2) == 3) ||
            (colors.dim() == 4 &&
             colors.sizes().slice(0, 2) == torch::IntArrayRef({C, N}) &&
             colors.size(3) == 3),
        "Invalid colors shape");
  } else {
    TORCH_CHECK((colors.dim() == 2 && colors.size(0) == N) ||
                    (colors.dim() == 3 &&
                     colors.sizes().slice(0, 2) == torch::IntArrayRef({C, N})),
                "Invalid colors shape");
  }

  // Project Gaussians to 2D
  bool cal_compensations = rasterize_mode == "antialiased";
  auto [camera_ids, gaussian_ids, radii, means2d, depths, conics,
        compensations] =
      fully_fused_projection(means, quats, scales, viewmats, Ks, width, height,
                             eps2d, near_plane, far_plane, radius_clip, packed,
                             sparse_grad, cal_compensations);

  auto proj_opacities = opacities.index({gaussian_ids});

  if (cal_compensations) {
    proj_opacities = proj_opacities * compensations;
  }

  // # global camera_ids
  meta["camera_ids"] = camera_ids;
  // # local gaussian_ids
  meta["gaussian_ids"] = gaussian_ids;
  meta["radii"] = radii;
  meta["means2d"] = means2d;
  meta["depths"] = depths;
  meta["conics"] = conics;
  meta["opacities"] = proj_opacities;

  // Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
  torch::Tensor proj_colors;
  if (!sh_degree.has_value()) {
    if (colors.dim() == 2) {
      proj_colors = colors.index({gaussian_ids});
    } else {
      proj_colors = colors.index({camera_ids, gaussian_ids});
    }
  } else {
    auto camtoworlds = torch::inverse(viewmats);

    auto dirs =
        means.index({gaussian_ids, torch::indexing::Slice()}) -
        camtoworlds.index({camera_ids, torch::indexing::Slice(0, 3), 3});
    auto masks = radii > 0;
    torch::Tensor shs;
    if (colors.dim() == 3) {
      shs = colors.index(
          {gaussian_ids, torch::indexing::Slice(), torch::indexing::Slice()});
    } else {
      shs = colors.index({camera_ids, gaussian_ids, torch::indexing::Slice(),
                          torch::indexing::Slice()});
    }
    proj_colors = spherical_harmonics(sh_degree.value(), dirs, shs, masks);

    proj_colors = torch::clamp_min(proj_colors + 0.5, 0.0);
  }

  // Rasterize to pixels
  torch::Tensor render_colors, render_alphas;
  if (render_mode == "RGB+D" || render_mode == "RGB+ED") {
    proj_colors = torch::cat({proj_colors, depths.unsqueeze(-1)}, -1);
    if (backgrounds.has_value()) {
      backgrounds = torch::cat(
          {backgrounds.value(), torch::zeros({C, 1}, backgrounds->device())},
          -1);
    }
  } else if (render_mode == "D" || render_mode == "ED") {
    proj_colors = depths.unsqueeze(-1);
    if (backgrounds.has_value()) {
      backgrounds = torch::zeros({C, 1}, backgrounds->device());
    }
  }

  auto tile_width =
      static_cast<int>(std::ceil(width / static_cast<float>(tile_size)));
  auto tile_height =
      static_cast<int>(std::ceil(height / static_cast<float>(tile_size)));
  auto [tiles_per_gauss, isect_ids, flatten_ids] =
      isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height,
                  true, packed, C, camera_ids, gaussian_ids);
  auto isect_offsets =
      isect_offset_encode(isect_ids, C, tile_width, tile_height);

  meta["tile_width"] = torch::tensor({tile_width});
  meta["tile_height"] = torch::tensor({tile_height});
  meta["tiles_per_gauss"] = tiles_per_gauss;
  meta["isect_offsets"] = isect_offsets;
  meta["width"] = torch::tensor({width});
  meta["height"] = torch::tensor({height});
  meta["n_cameras"] = torch::tensor({C});

  /* meta.update(
        {
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
              "tile_size": tile_size,
        }
    ) */

  auto means2d_absgrad =
      absgrad ? torch::zeros_like(means2d).requires_grad_() : torch::Tensor();

  if (proj_colors.size(-1) > channel_chunk) {
    int n_chunks = (proj_colors.size(-1) + channel_chunk - 1) / channel_chunk;
    std::vector<torch::Tensor> render_colors_vec, render_alphas_vec;
    for (int i = 0; i < n_chunks; ++i) {
      auto colors_chunk =
          proj_colors.slice(-1, i * channel_chunk, (i + 1) * channel_chunk);
      auto backgrounds_chunk =
          backgrounds.has_value()
              ? at::optional<torch::Tensor>(backgrounds->slice(
                    -1, i * channel_chunk, (i + 1) * channel_chunk))
              : at::nullopt;
      auto [render_colors_, render_alphas_] = rasterize_to_pixels(
          means2d, conics, colors_chunk, proj_opacities, width, height,
          tile_size, isect_offsets, flatten_ids, backgrounds_chunk, at::nullopt,
          packed, means2d_absgrad);
      render_colors_vec.push_back(render_colors_);
      render_alphas_vec.push_back(render_alphas_);
    }
    render_colors = torch::cat(render_colors_vec, -1);
    render_alphas = render_alphas_vec[0];
  } else {
    std::tie(render_colors, render_alphas) =
        rasterize_to_pixels(means2d, conics, proj_colors, proj_opacities, width,
                            height, tile_size, isect_offsets, flatten_ids,
                            backgrounds, at::nullopt, packed, means2d_absgrad);
  }
  if (absgrad) {
    meta["absgrad"] = means2d_absgrad;
  }

  if (render_mode == "ED" || render_mode == "RGB+ED") {
    render_colors = torch::cat(
        {render_colors.slice(-1, 0, -1),
         render_colors.slice(-1, -1) / render_alphas.clamp_min(1e-10f)},
        -1);
  }

  // std::cout << "render_colors.max(): " << render_colors.slice(-1, 0, 3).max()
  //           << std::endl;

  return std::make_tuple(render_colors, render_alphas, meta);
}
} // namespace gsplat_cpp