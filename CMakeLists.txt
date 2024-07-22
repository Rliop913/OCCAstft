cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(occaSTFT VERSION 0.1.0 LANGUAGES C CXX)

find_package(OCCA REQUIRED)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(extern/googletest)

enable_testing()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

set(srcs
    ${CMAKE_SOURCE_DIR}/src/STFT.cpp
)

set(test_dummy
    ${CMAKE_SOURCE_DIR}/src/main.cpp    
    ${CMAKE_SOURCE_DIR}/miniaudio.c

)

set(testSrcs
    ${CMAKE_SOURCE_DIR}/src/tests/tt.cpp)

add_executable(occaSTFT ${srcs} ${test_dummy})
add_executable(testerOcca ${testSrcs} ${srcs})
add_test(NAME OccaTest COMMAND testerOcca)
target_link_libraries(occaSTFT PUBLIC OCCA::libocca)
target_link_libraries(testerOcca PRIVATE gtest_main gtest OCCA::libocca)
