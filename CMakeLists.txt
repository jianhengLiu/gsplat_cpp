cmake_minimum_required(VERSION 3.18)
project(gsplat_cpp)

# Ensure `-fPIC` is used
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define CUDA flags
set(NVCC_FLAGS "-O3 --use_fast_math")

# Set up CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS}")

find_package(Torch REQUIRED)

include_directories(gsplat_cpp submodules/gsplat
                    submodules/gsplat/gsplat/cuda/csrc/third_party/glm)

file(GLOB_RECURSE GSPLAT_SOURCES submodules/gsplat/gsplat/cuda/*.cu)

add_library(gsplat ${GSPLAT_SOURCES})
target_link_libraries(gsplat ${TORCH_LIBRARIES})

file(GLOB_RECURSE GSPLAT_CPP_SOURCES gsplat_cpp/*.cpp)
add_library(gsplat_cpp ${GSPLAT_CPP_SOURCES})
target_link_libraries(gsplat_cpp gsplat)
