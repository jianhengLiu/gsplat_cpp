#include "rendering.h"
#include <map>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <tuple>

#include "fully_fused_projection.hpp"
#include "isect_tiles.hpp"
#include "rasterize_to_pixels.hpp"
#include "spherical_harmonics.hpp"

std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>>
rasterization(torch::Tensor &means, torch::Tensor &quats, torch::Tensor &scales,
              torch::Tensor &opacities, torch::Tensor &colors,
              const torch::Tensor &viewmats, const torch::Tensor &Ks, int width,
              int height, float near_plane, float far_plane, float radius_clip,
              float eps2d, std::optional<int> sh_degree, bool packed,
              int tile_size, at::optional<torch::Tensor> backgrounds,
              const std::string &render_mode, bool sparse_grad, bool absgrad,
              const std::string &rasterize_mode, int channel_chunk,
              bool distributed, const std::string &camera_model) {
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
              "Invalid opacities shape");
  TORCH_CHECK(viewmats.sizes() == torch::IntArrayRef({C, 4, 4}),
              "Invalid viewmats shape");
  TORCH_CHECK(Ks.sizes() == torch::IntArrayRef({C, 3, 3}), "Invalid Ks shape");
  TORCH_CHECK(render_mode == "RGB" || render_mode == "D" ||
                  render_mode == "ED" || render_mode == "RGB+D" ||
                  render_mode == "RGB+ED",
              "Invalid render_mode");

  /* def reshape_view(C: int, world_view: torch.Tensor, N_world: list) ->
  torch.Tensor: view_list = list( map( lambda x: x.split(int(x.shape[0] / C),
  dim=0), world_view.split([C * N_i for N_i in N_world], dim=0),
          )
      )
      return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

  if sh_degree is None:
      # treat colors as post-activation values, should be in shape [N, D] or [C,
  N, D] assert (colors.dim() == 2 and colors.shape[0] == N) or ( colors.dim() ==
  3 and colors.shape[:2] == (C, N)
      ), colors.shape
      if distributed:
          assert (
              colors.dim() == 2
          ), "Distributed mode only supports per-Gaussian colors."
  else:
      # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N,
  K, 3] # Allowing for activating partial SH bands assert ( colors.dim() == 3
  and colors.shape[0] == N and colors.shape[2] == 3 ) or ( colors.dim() == 4 and
  colors.shape[:2] == (C, N) and colors.shape[3] == 3
      ), colors.shape
      assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
      if distributed:
          assert (
              colors.dim() == 3
          ), "Distributed mode only supports per-Gaussian colors."

  if absgrad:
      assert not distributed, "AbsGrad is not supported in distributed mode."

  # If in distributed mode, we distribute the projection computation over
  Gaussians # and the rasterize computation over cameras. So first we gather the
  cameras # from all ranks for projection. if distributed: world_rank =
  torch.distributed.get_rank() world_size = torch.distributed.get_world_size()

      # Gather the number of Gaussians in each rank.
      N_world = all_gather_int32(world_size, N, device=device)

      # Enforce that the number of cameras is the same across all ranks.
      C_world = [C] * world_size
      viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])

      # Silently change C from local #Cameras to global #Cameras.
      C = len(viewmats) */

  // Project Gaussians to 2D
  auto [camera_ids, gaussian_ids, radii, means2d, depths, conics,
        compensations] =
      fully_fused_projection(means, quats, scales, viewmats, Ks, width, height,
                             eps2d, packed, near_plane, far_plane, radius_clip,
                             sparse_grad, (rasterize_mode == "antialiased"));

  opacities = opacities.index({gaussian_ids});

  if (compensations.defined()) {
    opacities = opacities * compensations;
  }

  // # global camera_ids
  meta["camera_ids"] = camera_ids;
  // # local gaussian_ids
  meta["gaussian_ids"] = gaussian_ids;
  meta["radii"] = radii;
  meta["means2d"] = means2d;
  meta["depths"] = depths;
  meta["conics"] = conics;
  meta["opacities"] = opacities;

  // Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
  if (!sh_degree.has_value()) {
    if (colors.dim() == 2) {
      colors = colors.index({gaussian_ids});
    } else {
      colors = colors.index({camera_ids, gaussian_ids});
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
    colors = spherical_harmonics(sh_degree.value(), dirs, shs, masks);

    colors = torch::clamp_min(colors + 0.5, 0.0);
  }

  // Rasterize to pixels
  torch::Tensor render_colors, render_alphas;
  if (render_mode == "RGB+D" || render_mode == "RGB+ED") {
    colors = torch::cat({colors, depths.unsqueeze(-1)}, -1);
    if (backgrounds.has_value()) {
      backgrounds = torch::cat(
          {backgrounds.value(), torch::zeros({C, 1}, backgrounds->device())},
          -1);
    }
  } else if (render_mode == "D" || render_mode == "ED") {
    colors = depths.unsqueeze(-1);
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
      isect_offset_encode(tiles_per_gauss, C, tile_width, tile_height);

  meta["tile_width"] = torch::tensor({tile_width});
  meta["tile_height"] = torch::tensor({tile_height});
  meta["tiles_per_gauss"] = tiles_per_gauss;
  meta["isect_offsets"] = isect_offsets;

  /* meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
    ) */

  if (colors.size(-1) > channel_chunk) {
    int n_chunks = (colors.size(-1) + channel_chunk - 1) / channel_chunk;
    std::vector<torch::Tensor> render_colors_vec, render_alphas_vec;
    for (int i = 0; i < n_chunks; ++i) {
      auto colors_chunk = colors.index(
          {torch::indexing::Slice(), torch::indexing::Slice(),
           torch::indexing::Slice(),
           torch::indexing::Slice(i * channel_chunk, (i + 1) * channel_chunk)});
      auto backgrounds_chunk =
          backgrounds.has_value()
              ? backgrounds->index(
                    {torch::indexing::Slice(),
                     torch::indexing::Slice(i * channel_chunk,
                                            (i + 1) * channel_chunk)})
              : torch::Tensor();
      auto [render_colors_, render_alphas_] = rasterize_to_pixels(
          means2d, conics, colors_chunk, opacities, width, height, tile_size,
          isect_offsets, flatten_ids, backgrounds_chunk, torch::Tensor(),
          packed, absgrad);
      render_colors_vec.push_back(render_colors_);
      render_alphas_vec.push_back(render_alphas_);
    }
    render_colors = torch::cat(render_colors_vec, -1);
    render_alphas = render_alphas_vec[0];
  } else {
    std::tie(render_colors, render_alphas) =
        rasterize_to_pixels(means2d, conics, colors, opacities, width, height,
                            tile_size, isect_offsets, flatten_ids, backgrounds,
                            torch::Tensor(), packed, absgrad);
  }

  if (render_mode == "ED" || render_mode == "RGB+ED") {
    render_colors = torch::cat(
        {render_colors.index(
             {torch::indexing::Slice(), torch::indexing::Slice(),
              torch::indexing::Slice(), torch::indexing::Slice(0, -1)}),
         render_colors.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(), -1}) /
             render_alphas.clamp_min(1e-10)},
        -1);
  }

  return std::make_tuple(render_colors, render_alphas, meta);
}