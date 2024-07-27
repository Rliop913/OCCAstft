cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(openclSTFT VERSION 0.1.0 LANGUAGES C CXX)

include(OpenCL)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(openclsrc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/STFT.cpp)


add_executable(openclRun ${openclsrc})

target_link_libraries(openclRun OpenCL::OpenCL)

