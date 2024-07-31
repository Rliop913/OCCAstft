cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(cudaSTFT VERSION 0.1.0 LANGUAGES C CXX)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(cudasrc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/)


add_library(cudaRun cudasrc)

target_link_libraries(cudaRun PRIVATE CUDA::cuda_driver)
#ptx 변환작업 필요
#libcuda만 사용, 즉 드라이버 API만 사용