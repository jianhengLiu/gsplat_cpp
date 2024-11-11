cmake_minimum_required(VERSION 3.18)
project(gsplat_cpp)

set(CMAKE_CXX_STANDARD 17)
# Ensure `-fPIC` is used
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-sign-compare")

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Define CUDA flags
set(NVCC_FLAGS "-O3 --use_fast_math --expt-relaxed-constexpr")

# Set up CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS}")

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)

file(GLOB_RECURSE GSPLAT_SOURCES submodules/gsplat/gsplat/cuda/csrc/*.cu)

add_library(gsplat ${GSPLAT_SOURCES})
target_link_libraries(gsplat ${TORCH_LIBRARIES})
target_include_directories(
  gsplat PRIVATE ${PYTHON_INCLUDE_DIRS} submodules/gsplat/gsplat/cuda/include
                submodules/gsplat/gsplat/cuda/csrc/third_party/glm)

file(GLOB_RECURSE GSPLAT_CPP_SOURCES gsplat_cpp/*.cpp)
add_library(gsplat_cpp ${GSPLAT_CPP_SOURCES})
target_link_libraries(gsplat_cpp gsplat)
target_include_directories(gsplat_cpp PUBLIC submodules/gsplat)
