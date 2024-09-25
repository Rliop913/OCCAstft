cmake_minimum_required(VERSION 3.5.0)
#set(CMAKE_C_COMPILER clang)
#set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(occaSample VERSION 1.0.0 LANGUAGES C CXX)

find_package(OCCA REQUIRED)

add_executable(sample occaSample.cpp)

target_link_directories(sample PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(sample PRIVATE OCCA::libocca)