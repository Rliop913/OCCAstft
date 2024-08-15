cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(openclSTFT VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)


FetchContent_Declare(
  ixws
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  GIT_TAG v11.4.5
)

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

FetchContent_MakeAvailable(ixws)
FetchContent_MakeAvailable(opencl_headers)
FetchContent_MakeAvailable(opencl_hpp)


find_package(OpenCL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(CapnProto REQUIRED)
set(openclsrc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CLImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CL_Wrapper/src/cl_FACADE.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CL_Wrapper/src/cl_global_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL/RunnerCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct_SetGet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTcapnp.capnp.c++
    )


add_executable(openclRun ${openclsrc})
target_link_libraries(openclRun PRIVATE CapnProto::capnp CapnProto::kj)
message(${opencl_headers_SOURCE_DIR})

target_include_directories(openclRun PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL#reverseOOP
    ${CMAKE_CURRENT_SOURCE_DIR}/CL_Wrapper/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${opencl_hpp_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
message(${ixws_BINARY_DIR})
if(WIN32)
    if(MSVC)
        target_link_libraries(openclRun PRIVATE
            OpenCL::OpenCL
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
    else()
        target_link_libraries(openclRun PRIVATE 
        OpenCL::OpenCL
        ${ixws_BINARY_DIR}/ixwebsocket.lib)
    endif()
else()
    target_link_libraries(openclRun PRIVATE 
    OpenCL::OpenCL
    ${ixws_BINARY_DIR}/libixwebsocket.a)
    
endif()
target_link_libraries(openclRun PRIVATE ZLIB::ZLIB)
target_include_directories(openclRun PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)