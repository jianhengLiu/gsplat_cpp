#pragma once

#include <torch/torch.h>
#include <tuple>

namespace gsplat_cpp {

torch::Tensor
get_view_colors(const torch::Tensor &viewmats, const torch::Tensor &means,
                const torch::Tensor &radii,
                const torch::Tensor &colors, //[(C,) N, D] or [(C,) N, K, 3]
                const torch::Tensor &camera_ids,
                const torch::Tensor &gaussian_ids, at::optional<int> sh_degree);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
tile_encode(const int &width, const int &height, const int &tile_size,
            const torch::Tensor &means2d, const torch::Tensor &radii,
            const torch::Tensor &depths, const bool &packed,
            const int &camera_num, const torch::Tensor &camera_ids,
            const torch::Tensor &gaussian_ids);

std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>>
rasterization(
    const torch::Tensor &means,     //[N, 3]
    const torch::Tensor &quats,     // [N, 4]
    const torch::Tensor &scales,    // [N, 3]
    const torch::Tensor &opacities, // [N]
    const torch::Tensor &colors,    //[(C,) N, D] or [(C,) N, K, 3]
    const torch::Tensor &viewmats,  //[C, 4, 4]
    const torch::Tensor &Ks,        //[C, 3, 3]
    int width, int height,
    const std::string &render_mode =
        "RGB", //["RGB", "D", "ED", "RGB+D", "RGB+ED"]
    float near_plane = 0.01f, float far_plane = 1e10f, float radius_clip = 0.0f,
    float eps2d = 0.3f, at::optional<int> sh_degree = at::nullopt,
    bool packed = true, int tile_size = 16,
    at::optional<torch::Tensor> backgrounds = at::nullopt,
    bool sparse_grad = false, bool absgrad = false,
    const std::string &rasterize_mode = "classic", //"classic", "antialiased"
    int channel_chunk = 32);

std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>>
rasterization_2dgs(const torch::Tensor &means,     //[N, 3]
                   const torch::Tensor &quats,     // [N, 4]
                   const torch::Tensor &scales,    // [N, 3]
                   const torch::Tensor &opacities, // [N]
                   const torch::Tensor &colors, //[(C,) N, D] or [(C,) N, K, 3]
                   const torch::Tensor &viewmats, //[C, 4, 4]
                   const torch::Tensor &Ks,       //[C, 3, 3]
                   int width, int height,
                   const std::string &render_mode =
                       "RGB", //["RGB", "D", "ED", "RGB+D", "RGB+ED"]
                   float near_plane = 0.01f, float far_plane = 1e10f,
                   float radius_clip = 0.0f,
                   at::optional<int> sh_degree = at::nullopt,
                   bool packed = true, int tile_size = 16,
                   at::optional<torch::Tensor> backgrounds = at::nullopt,
                   bool sparse_grad = false, bool absgrad = false,
                   bool distloss = false,
                   const std::vector<torch::Tensor> &attributes = {});
} // namespace gsplat_cpp