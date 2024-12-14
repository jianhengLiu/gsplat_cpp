#include "rendering.h"
#include <map>
#include <string>

#include "fully_fused_projection.h"
#include "isect_tiles.hpp"
#include "rasterize_to_pixels.h"
#include "spherical_harmonics.hpp"
namespace gsplat_cpp {

torch::Tensor
get_view_colors(const torch::Tensor &viewmats, const torch::Tensor &means,
                const torch::Tensor &radii,
                const torch::Tensor &colors, //[(C,) N, D] or [(C,) N, K, 3]
                const torch::Tensor &camera_ids,
                const torch::Tensor &gaussian_ids,
                at::optional<int> sh_degree) {
  // Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
  torch::Tensor pt_colors;
  if (!sh_degree.has_value()) {
    if (colors.dim() == 2) {
      pt_colors = colors.index({gaussian_ids});
    } else {
      pt_colors = colors.index({camera_ids, gaussian_ids});
    }
  } else {
    auto camtoworlds = torch::inverse(viewmats);

    auto dirs =
        means.index({gaussian_ids, torch::indexing::Slice()}) -
        camtoworlds.index({camera_ids, torch::indexing::Slice(0, 3), 3});

    torch::Tensor shs;
    if (colors.dim() == 3) {
      shs = colors.index(
          {gaussian_ids, torch::indexing::Slice(), torch::indexing::Slice()});
    } else {
      shs = colors.index({camera_ids, gaussian_ids, torch::indexing::Slice(),
                          torch::indexing::Slice()});
    }
    pt_colors = spherical_harmonics(sh_degree.value(), dirs, shs, radii > 0);

    pt_colors = torch::clamp_min(pt_colors + 0.5, 0.0);
  }
  return pt_colors;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tile_encode(const int &width, const int &height, const int &tile_size,
            const torch::Tensor &means2d, const torch::Tensor &radii,
            const torch::Tensor &depths, const bool &packed,
            const int &camera_num, const torch::Tensor &camera_ids,
            const torch::Tensor &gaussian_ids) {
  auto tile_width = (int)(std::ceil(width / (float)tile_size));
  auto tile_height = (int)(std::ceil(height / (float)tile_size));
  auto [tiles_per_gauss, isect_ids, flatten_ids] =
      isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height,
                  true, packed, camera_num, camera_ids, gaussian_ids);
  auto isect_offsets =
      isect_offset_encode(isect_ids, camera_num, tile_width, tile_height);
  return {isect_offsets, flatten_ids, isect_offsets};
}

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
  // static auto p_t_pre = llog::CreateTimer("pre");
  // p_t_pre->tic();
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

  // p_t_pre->toc_sum();
  // static auto p_t_proj = llog::CreateTimer("proj");
  // p_t_proj->tic();

  // Project Gaussians to 2D
  bool cal_compensations = rasterize_mode == "antialiased";
  auto [camera_ids, gaussian_ids, radii, means2d, depths, conics,
        compensations] =
      fully_fused_projection(means, quats, scales, viewmats, Ks, width, height,
                             eps2d, near_plane, far_plane, radius_clip, packed,
                             sparse_grad, cal_compensations);

  auto pt_opacities = opacities.index({gaussian_ids});

  if (cal_compensations) {
    pt_opacities = pt_opacities * compensations;
  }

  // p_t_proj->toc_sum();
  // static auto p_t_sh = llog::CreateTimer("sh");
  // p_t_sh->tic();

  torch::Tensor pt_colors = get_view_colors(
      viewmats, means, radii, colors, camera_ids, gaussian_ids, sh_degree);

  // p_t_sh->toc_sum();
  // static auto p_t_tile = llog::CreateTimer("tile");
  // p_t_tile->tic();

  auto [tiles_per_gauss, flatten_ids, isect_offsets] =
      tile_encode(width, height, tile_size, means2d, radii, depths, packed, C,
                  camera_ids, gaussian_ids);

  // p_t_tile->toc_sum();
  // static auto p_t_ras = llog::CreateTimer("ras");
  // p_t_ras->tic();

  // Rasterize to pixels
  torch::Tensor render_colors, render_alphas;
  if (render_mode == "RGB+D" || render_mode == "RGB+ED") {
    pt_colors = torch::cat({pt_colors, depths.unsqueeze(-1)}, -1);
    if (backgrounds.has_value()) {
      backgrounds = torch::cat(
          {backgrounds.value(), torch::zeros({C, 1}, backgrounds->device())},
          -1);
    }
  } else if (render_mode == "D" || render_mode == "ED") {
    pt_colors = depths.unsqueeze(-1);
    if (backgrounds.has_value()) {
      backgrounds = torch::zeros({C, 1}, backgrounds->device());
    }
  }

  auto means2d_absgrad = torch::zeros_like(means2d).requires_grad_(absgrad);

  if (pt_colors.size(-1) > channel_chunk) {
    int n_chunks = (pt_colors.size(-1) + channel_chunk - 1) / channel_chunk;
    std::vector<torch::Tensor> render_colors_vec, render_alphas_vec;
    for (int i = 0; i < n_chunks; ++i) {
      auto colors_chunk =
          pt_colors.slice(-1, i * channel_chunk, (i + 1) * channel_chunk);
      auto backgrounds_chunk =
          backgrounds.has_value()
              ? at::optional<torch::Tensor>(backgrounds->slice(
                    -1, i * channel_chunk, (i + 1) * channel_chunk))
              : at::nullopt;
      auto [render_colors_, render_alphas_] = rasterize_to_pixels(
          means2d, conics, colors_chunk, pt_opacities, width, height, tile_size,
          isect_offsets, flatten_ids, backgrounds_chunk, at::nullopt, packed,
          means2d_absgrad);
      render_colors_vec.push_back(render_colors_);
      render_alphas_vec.push_back(render_alphas_);
    }
    render_colors = torch::cat(render_colors_vec, -1);
    render_alphas = render_alphas_vec[0];
  } else {
    std::tie(render_colors, render_alphas) =
        rasterize_to_pixels(means2d, conics, pt_colors, pt_opacities, width,
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
  // p_t_ras->toc_sum();
  // static auto p_t_meta = llog::CreateTimer("meta");
  // p_t_meta->tic();

  // # global camera_ids
  meta["camera_ids"] = camera_ids;
  // # local gaussian_ids
  meta["gaussian_ids"] = gaussian_ids;
  meta["radii"] = radii;
  meta["means2d"] = means2d;
  meta["depths"] = depths;
  meta["conics"] = conics;
  meta["opacities"] = pt_opacities;
  // meta["tile_width"] = torch::tensor({tile_width});
  // meta["tile_height"] = torch::tensor({tile_height});
  meta["tiles_per_gauss"] = tiles_per_gauss;
  meta["isect_offsets"] = isect_offsets;
  meta["width"] = torch::tensor({width});
  meta["height"] = torch::tensor({height});
  meta["n_cameras"] = torch::tensor({C});
  // p_t_meta->toc_sum();
  return std::make_tuple(render_colors, render_alphas, meta);
}

std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>>
rasterization_2dgs(const torch::Tensor &means,     //[N, 3]
                   const torch::Tensor &quats,     // [N, 4]
                   const torch::Tensor &scales,    // [N, 3]
                   const torch::Tensor &opacities, // [N]
                   const torch::Tensor &colors, //[(C,) N, D] or [(C,) N, K, 3]
                   const torch::Tensor &viewmats, //[C, 4, 4]
                   const torch::Tensor &Ks,       //[C, 3, 3]
                   int width, int height, const std::string &render_mode,
                   float near_plane, float far_plane, float radius_clip,
                   at::optional<int> sh_degree, bool packed, int tile_size,
                   at::optional<torch::Tensor> backgrounds, bool sparse_grad,
                   bool absgrad, bool distloss,
                   const std::vector<torch::Tensor> &attributes) {
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
        (colors.dim() == 3 && colors.size(0) == N && colors.size(2) == 3),
        "Invalid colors shape");
    TORCH_CHECK((std::pow(sh_degree.value() + 1, 2) <= colors.size(1)),
                "Invalid colors shape");
  } else {
    TORCH_CHECK((colors.dim() == 2 && colors.size(0) == N) ||
                    (colors.dim() == 3 &&
                     colors.sizes().slice(0, 2) == torch::IntArrayRef({C, N})),
                "Invalid colors shape");
  }

  // p_t_pre->toc_sum();
  // static auto p_t_proj = llog::CreateTimer("proj");
  // p_t_proj->tic();

  // # Compute Ray-Splat intersection transformation.
  auto [camera_ids, gaussian_ids, radii, means2d, depths, ray_transforms,
        normals, samples, samples_weights] =
      fully_fused_projection_2dgs(means, quats, scales, viewmats, Ks, width,
                                  height, near_plane, far_plane, radius_clip,
                                  packed, sparse_grad);
  auto pt_opacities = opacities.index({gaussian_ids});

  // p_t_proj->toc_sum();
  // static auto p_t_sh = llog::CreateTimer("sh");
  // p_t_sh->tic();

  torch::Tensor pt_colors = get_view_colors(
      viewmats, means, radii, colors, camera_ids, gaussian_ids, sh_degree);

  // p_t_sh->toc_sum();
  // static auto p_t_tile = llog::CreateTimer("tile");
  // p_t_tile->tic();

  auto [tiles_per_gauss, flatten_ids, isect_offsets] =
      tile_encode(width, height, tile_size, means2d, radii, depths, packed, C,
                  camera_ids, gaussian_ids);

  // p_t_tile->toc_sum();
  // static auto p_t_ras = llog::CreateTimer("ras");
  // p_t_ras->tic();

  // Rasterize to pixels
  if (render_mode == "RGB+D" || render_mode == "RGB+ED") {
    pt_colors = torch::cat({pt_colors, depths.unsqueeze(-1)}, -1);
  } else if (render_mode == "D" || render_mode == "ED") {
    pt_colors = depths.unsqueeze(-1);
  }

  int attri_channels = 0;
  for (const auto &atrribute : attributes) {
    if (atrribute.defined()) {
      pt_colors =
          torch::cat({pt_colors, atrribute.index_select(0, gaussian_ids)}, -1);
      attri_channels += atrribute.size(1);
    }
  }

  auto means2d_absgrad = torch::zeros_like(means2d).requires_grad_(absgrad);
  auto densify =
      torch::zeros_like(means2d, means.options().requires_grad(true));
  auto [render_colors, render_depths, render_alphas, render_normals,
        render_distort, render_median] =
      rasterize_to_pixels_2dgs(means2d, ray_transforms, pt_colors, pt_opacities,
                               normals, densify, width, height, tile_size,
                               isect_offsets, flatten_ids, backgrounds,
                               at::nullopt, packed, means2d_absgrad, distloss);

  if (absgrad) {
    meta["absgrad"] = means2d_absgrad;
  }

  if (render_mode == "ED" || render_mode == "RGB+ED") {
    if (attri_channels > 0) {
      // TODO: adapt for multi attributes
      // render_colors = torch::cat(
      //     {render_colors.slice(-1, 0, 3),
      //      render_colors.slice(-1, 3, 4) / render_alphas.clamp_min(1e-10f),
      //      render_colors.slice(-1, 4)},
      //     -1);
      render_colors =
          torch::cat({render_colors.slice(-1, 0, 3),
                      render_depths / render_alphas.clamp_min(
                                          1e-10f), // only for mean depth
                      render_colors.slice(-1, 4)},
                     -1);
    } else {
      render_colors = torch::cat(
          {render_colors, render_depths / render_alphas.clamp_min(1e-10f)},
          -1); // only for mean depth
    }
  }

  // transform normal to world space
  render_normals =
      render_normals.matmul(viewmats.inverse()
                                .index({0, torch::indexing::Slice(0, 3),
                                        torch::indexing::Slice(0, 3)})
                                .t());

  // p_t_ras->toc_sum();
  // static auto p_t_meta = llog::CreateTimer("meta");
  // p_t_meta->tic();

  meta["render_normal"] = render_normals;
  meta["render_median"] = render_median;

  // gs normal in camera space
  meta["normal"] = normals;
  // # global camera_ids
  meta["camera_ids"] = camera_ids;
  // # local gaussian_ids
  meta["gaussian_ids"] = gaussian_ids;
  meta["radii"] = radii;
  meta["means2d"] = means2d;
  meta["gradient_2dgs"] =
      densify; // This holds the gradient used for densification for 2dgs
  meta["depths"] = depths;
  meta["opacity"] = opacities;

  // meta["tile_width"] = torch::tensor({tile_width});
  // meta["tile_height"] = torch::tensor({tile_height});
  meta["tiles_per_gauss"] = tiles_per_gauss;
  meta["isect_offsets"] = isect_offsets;
  meta["width"] = torch::tensor({width});
  meta["height"] = torch::tensor({height});
  meta["n_cameras"] = torch::tensor({C});

  // p_t_meta->toc_sum();
  return std::make_tuple(render_colors, render_alphas, meta);
}
} // namespace gsplat_cpp