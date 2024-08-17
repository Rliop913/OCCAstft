cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(occaSTFT VERSION 0.1.0 LANGUAGES C CXX)

# add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
# set(CMAKE_BUILD_TYPE Debug)

include(FetchContent)
if(WIN32)#window
add_definitions(-DOS_WINDOWS)
add_definitions(-DWIN32_LEAN_AND_MEAN)
# set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};G:/vcpkg/installed/x64-windows")
# set(CapnProto_DIR "G:/vcpkg/installed/x64-windows/share/capnproto")
# set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
# set(VCPKG_TARGET_TRIPLET "x64-windows")
else()
add_definitions(-DOS_POSIX)


endif()# linux
# if(MSVC)
#     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
# endif()

# set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "" FORCE)
# set(USE_ZLIB ON CACHE BOOL "" FORCE)
# set(USE_TLS OFF CACHE BOOL "" FORCE)
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
include(CTest)
include(Catch)
enable_testing()
# find_package(ixwebsocket REQUIRED)
find_package(CUDA QUIET)
find_package(OpenCL)
find_package(OpenMP)
find_package(CapnProto REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")



set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)



include_directories(
    #${CMAKE_SOURCE_DIR}/OCCA/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
    #${ixws_SOURCE_DIR}
)

set(srcs
    ${CMAKE_SOURCE_DIR}/src/STFT.cpp
)

set(test_dummy
    ${CMAKE_SOURCE_DIR}/src/template.cpp    
    ${CMAKE_SOURCE_DIR}/miniaudio.c

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
message(${ixws_BINARY_DIR})
add_executable(STFTMain ${mainsrc})


add_executable(TEST_STFT ${mainsrc})

target_link_libraries(STFTMain PRIVATE CapnProto::capnp CapnProto::kj)
target_link_libraries(TEST_STFT PRIVATE CapnProto::capnp CapnProto::kj)



if(WIN32)
target_link_libraries(STFTMain PRIVATE ws2_32)
target_link_libraries(TEST_STFT PRIVATE ws2_32)
    if(MSVC)
        target_link_libraries(STFTMain PRIVATE 
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
        target_link_libraries(TEST_STFT PRIVATE 
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
    else()
        target_link_libraries(STFTMain PRIVATE ${ixws_BINARY_DIR}/ixwebsocket.lib)
        target_link_libraries(TEST_STFT PRIVATE ${ixws_BINARY_DIR}/ixwebsocket.lib)
    endif()
    target_link_libraries(STFTMain PRIVATE ws2_32)
    target_link_libraries(TEST_STFT PRIVATE ws2_32)
else()
    target_link_libraries(STFTMain PRIVATE ${ixws_BINARY_DIR}/libixwebsocket.a)
    target_link_libraries(TEST_STFT PRIVATE ${ixws_BINARY_DIR}/libixwebsocket.a)
endif()

target_link_libraries(STFTMain PRIVATE ZLIB::ZLIB)
target_link_libraries(TEST_STFT PRIVATE ZLIB::ZLIB)
target_include_directories(TEST_STFT PRIVATE ${catch2_SOURCE_DIR})

target_include_directories(STFTMain PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)
target_include_directories(TEST_STFT PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)
target_link_libraries(TEST_STFT PRIVATE Catch2::Catch2WithMain)
#catch_discover_tests(TEST_STFT)
message(${ZLIB_LIBRARIES})
# gtest_discover_tests(TEST_STFT)
