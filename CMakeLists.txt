cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(occaSTFT VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)
if(WIN32)#window
add_definitions(-DOS_WINDOWS)
add_definitions(-DWIN32_LEAN_AND_MEAN)
else()
add_definitions(-DOS_POSIX)
find_package(OCCA REQUIRED)

endif()

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)

FetchContent_Declare(
  ixws
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  GIT_TAG v11.4.5
)

FetchContent_MakeAvailable(ixws)
FetchContent_MakeAvailable(Catch2)
find_package(ZLIB REQUIRED)
find_package(CUDA QUIET)
find_package(OpenCL)
find_package(OpenMP)
find_package(CapnProto REQUIRED)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

set(srcs
    ${CMAKE_SOURCE_DIR}/src/occa.cpp
)

set(testSrcs
    ${CMAKE_SOURCE_DIR}/test/test_fft_struct.cpp)



if(CUDA_FOUND)
add_subdirectory(${CMAKE_SOURCE_DIR}/cross_gpgpu/CUDA)
endif()

if(OpenCL_FOUND)
add_subdirectory(${CMAKE_SOURCE_DIR}/cross_gpgpu/OpenCL)
endif()

if(OpenMP_FOUND)
add_subdirectory(${CMAKE_SOURCE_DIR}/cross_gpgpu/OpenMP)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/cross_gpgpu/Serial)

set(mainsrc 
    ${CMAKE_SOURCE_DIR}/src/runtimeChecker.cpp
    ${CMAKE_SOURCE_DIR}/src/STFTProxy.cpp
    ${CMAKE_SOURCE_DIR}/src/FFTStruct.cpp
    ${CMAKE_SOURCE_DIR}/src/FFTcapnp.capnp.c++
    ${CMAKE_SOURCE_DIR}/src/FFTStruct_SetGet.cpp    
    )


add_library(STFTMain ${mainsrc})

target_link_libraries(STFTMain PRIVATE CapnProto::capnp CapnProto::kj)
# if(OCCA_FOUND)
# add_executable(TEST_STFT ${test_dummy})
# target_link_libraries(TEST_STFT PRIVATE CapnProto::capnp CapnProto::kj)
# else()
# add_library(TEST_STFT ${testSrcs})
# endif()



if(WIN32)
target_link_libraries(STFTMain PRIVATE ws2_32)
    if(MSVC)
        target_link_libraries(STFTMain PRIVATE 
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
    else()
        target_link_libraries(STFTMain PRIVATE ${ixws_BINARY_DIR}/ixwebsocket.lib)
    endif()
    target_link_libraries(STFTMain PRIVATE ws2_32)
else()
    target_link_libraries(STFTMain PRIVATE ${ixws_BINARY_DIR}/libixwebsocket.a)
endif()

target_link_libraries(STFTMain PRIVATE ZLIB::ZLIB)

target_include_directories(STFTMain PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)

# if(OCCA_FOUND)
# target_link_libraries(TEST_STFT PRIVATE ${CMAKE_SOURCE_DIR}/libocca.so)
# target_include_directories(TEST_STFT PRIVATE ${OCCA_INCLUDE_DIR})
# endif()