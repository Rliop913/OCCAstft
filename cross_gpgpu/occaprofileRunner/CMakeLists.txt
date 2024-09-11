cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(occaprofileSTFT VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)


FetchContent_Declare(
  ixws
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  GIT_TAG v11.4.5
)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)



FetchContent_MakeAvailable(ixws)

FetchContent_MakeAvailable(json)
if(CapnProto_FOUND)
else()
find_package(CapnProto REQUIRED)
endif()
find_package(ZLIB REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(occasrc
${CMAKE_CURRENT_SOURCE_DIR}/src/CudaImpl.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL/RunnerCommon.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct_SetGet.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTcapnp.capnp.c++
)


add_executable(occaRun ${occasrc})
target_link_libraries(occaRun PRIVATE CapnProto::capnp CapnProto::kj)

target_link_libraries(occaRun PRIVATE CUDA::cuda_driver)
target_include_directories(occaRun PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(occaRun PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL#reverseOOP
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)


if(WIN32)
    target_link_libraries(occaRun PRIVATE ws2_32)
    if(MSVC)
        target_link_libraries(occaRun PRIVATE
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
    else()
        target_link_libraries(occaRun PRIVATE 
        ${ixws_BINARY_DIR}/ixwebsocket.lib)
    endif()
else()
    target_link_libraries(occaRun PRIVATE 
    ${ixws_BINARY_DIR}/libixwebsocket.a)
    
endif()


target_include_directories(occaRun PRIVATE ${json_SOURCE_DIR}/include)

target_link_libraries(occaRun PRIVATE ZLIB::ZLIB)
target_include_directories(occaRun PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)