cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(openclSTFT VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)

FetchContent_Declare(
  opencl_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
  GIT_TAG v2024.05.08
)

FetchContent_Declare(
  opencl_hpp
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
  GIT_TAG v2024.05.08
)

FetchContent_MakeAvailable(opencl_headers)
FetchContent_MakeAvailable(opencl_hpp)


find_package(OpenCL REQUIRED)

set(openclsrc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/STFT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CL_Wrapper/src/cl_FACADE.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CL_Wrapper/src/cl_global_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../miniaudio.c)


add_executable(openclRun ${openclsrc})
message(${opencl_headers_SOURCE_DIR})
target_include_directories(openclRun PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL#reverseOOP
    ${CMAKE_CURRENT_SOURCE_DIR}/CL_Wrapper/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${opencl_hpp_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../miniaudio.h
)
target_link_libraries(openclRun OpenCL::OpenCL)

