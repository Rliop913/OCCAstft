cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(occaSTFT VERSION 0.1.0 LANGUAGES C CXX)
# add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
set(CMAKE_BUILD_TYPE Release)
find_package(CUDA QUIET)
find_package(OpenCL)
find_package(OpenMP)


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
if(WIN32)#window



else()
find_package(OCCA REQUIRED)


endif()# linux

include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(extern/googletest)

enable_testing()

include_directories(
    #${CMAKE_SOURCE_DIR}/OCCA/include
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
if(WIN32)
#find_library(OccaLib occa.lib HINTS ${CMAKE_SOURCE_DIR}/OCCA/lib)
# target_link_libraries(occaSTFT PUBLIC ${CMAKE_SOURCE_DIR}/OCCA/lib/occa.lib)
# target_link_libraries(testerOcca PRIVATE gtest_main gtest ${OccaLib})
else()
target_link_libraries(occaSTFT PUBLIC OCCA::libocca)
target_link_libraries(testerOcca PRIVATE gtest_main gtest OCCA::libocca)
endif()





if(CUDA_FOUND)
project(cudaRun VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(cudasrc
    ${CMAKE_SOURCE_DIR}/cross_gpgpu/CUDA/src/STFT.cpp
)
add_executable(cudaRun ${cudasrc})
target_include_directories(cudaRun
    ${CMAKE_SOURCE_DIR}/cross_gpgpu/CUDA/include
)
target_link_libraries(cudaRun PRIVATE CUDA::CUDA)
endif()

if(OpenCL_FOUND)

# project(openclRun VERSION 0.1.0 LANGUAGES C CXX)
set(openclsrc
    ${CMAKE_SOURCE_DIR}/cross_gpgpu/OpenCL/src/STFT.cpp
    ${CMAKE_SOURCE_DIR}/cross_gpgpu/OpenCL/CL_Wrapper/src/cl_FACADE.cpp
    ${CMAKE_SOURCE_DIR}/cross_gpgpu/OpenCL/CL_Wrapper/src/cl_global_custom.cpp
)
add_executable(openclRun ${openclsrc})
target_include_directories(openclRun PRIVATE
    ${CMAKE_SOURCE_DIR}/cross_gpgpu/OpenCL/CL_Wrapper/include    
    ${CMAKE_SOURCE_DIR}/cross_gpgpu/OpenCL/include/
)
target_link_libraries(openclRun PRIVATE OpenCL::OpenCL)

endif()

if(OpenMP_FOUND)

endif()
