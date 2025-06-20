cmake_minimum_required(VERSION 3.18)
project(gsplat_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
# Ensure `-fPIC` is used
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -Wno-sign-compare -DAT_PARALLEL_OPENMP -fopenmp")

# Define CUDA flags
set(NVCC_FLAGS
    "-O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress 20012,186")

# Set up CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS}")

find_package(Torch REQUIRED)

# Separate CUDA source files from C++ source files
file(GLOB_RECURSE GSPLAT_SOURCES submodules/gsplat/gsplat/cuda/csrc/*.cpp
     submodules/gsplat/gsplat/cuda/csrc/*.cu)
# Filter out third_party files
list(FILTER GSPLAT_SOURCES EXCLUDE REGEX ".*third_party.*")

add_library(gsplat ${GSPLAT_SOURCES})
target_link_libraries(gsplat ${TORCH_LIBRARIES})
target_include_directories(
  gsplat PUBLIC submodules/gsplat/gsplat/cuda/include
                submodules/gsplat/gsplat/cuda/csrc/third_party/glm)

# Add your gsplat_cpp library
file(GLOB_RECURSE GSPLAT_CPP_SOURCES gsplat_cpp/*.cpp)
message(STATUS "GSPLAT_CPP_SOURCES: ${GSPLAT_CPP_SOURCES}")
add_library(gsplat_cpp ${GSPLAT_CPP_SOURCES})
target_link_libraries(gsplat_cpp gsplat)
target_include_directories(gsplat_cpp PUBLIC submodules/gsplat)
