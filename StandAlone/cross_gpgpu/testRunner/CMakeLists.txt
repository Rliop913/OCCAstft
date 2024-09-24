cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(templateRunner VERSION 0.1.0 LANGUAGES C CXX CUDA)




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

find_package(clFFT REQUIRED)
find_package(ZLIB REQUIRED)
if(NOT CUDA_FOUND)
find_package(CUDA REQUIRED)
endif()
if(CapnProto_FOUND)
else()
find_package(CapnProto REQUIRED)
endif()
find_package(OpenCL REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_library(CUFFT_LIBRARY cufft HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(testsrc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/testRunner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clfftImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cufftImpl.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL/RunnerCommon.cpp #common implementation
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct_SetGet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTcapnp.capnp.c++
    
    )



add_executable(testRunner ${testsrc})
target_include_directories(testRunner PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(testRunner PRIVATE CapnProto::capnp CapnProto::kj)
target_link_libraries(testRunner PRIVATE ${CUFFT_LIBRARY})
target_link_libraries(testRunner PRIVATE ${CUDA_LIBRARIES})

target_include_directories(testRunner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL#reverseOOP
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(WIN32)
    target_link_libraries(testRunner PRIVATE ws2_32)
    if(MSVC)
        target_link_libraries(testRunner PRIVATE
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
    else()
        target_link_libraries(testRunner PRIVATE 
        ${ixws_BINARY_DIR}/ixwebsocket.lib)
    endif()
else()
    target_link_libraries(testRunner PRIVATE 
    ${ixws_BINARY_DIR}/libixwebsocket.a)
    
endif()
target_link_libraries(testRunner PRIVATE ZLIB::ZLIB)
target_include_directories(testRunner PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)



target_include_directories(testRunner PRIVATE ${json_SOURCE_DIR}/include)

# get_cmake_property(_variableNames VARIABLES)
# foreach (_variableName ${_variableNames})
#   message(STATUS "${_variableName} = ${${_variableName}}")
# endforeach()



target_link_libraries(testRunner PRIVATE ${CLFFT_LIBRARIES})
target_link_libraries(testRunner PRIVATE OpenCL::OpenCL)


