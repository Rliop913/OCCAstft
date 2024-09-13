cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(openmpSTFT VERSION 0.1.0 LANGUAGES C CXX)


include(FetchContent)


FetchContent_Declare(
  ixws
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  GIT_TAG v11.4.5
)

FetchContent_MakeAvailable(ixws)

find_package(ZLIB REQUIRED)
if(CapnProto_FOUND)
else()
find_package(CapnProto REQUIRED)
endif()
find_package(OpenMP REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(openmpsrc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/OpenmpImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL/RunnerCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct_SetGet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTcapnp.capnp.c++
    )


add_executable(openmpRun ${openmpsrc})



target_link_libraries(openmpRun PRIVATE CapnProto::capnp CapnProto::kj)

target_include_directories(openmpRun PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL#reverseOOP
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(WIN32)
    target_link_libraries(openmpRun PRIVATE ws2_32)
    if(MSVC)
        target_link_libraries(openmpRun PRIVATE
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
    else()
        target_link_libraries(openmpRun PRIVATE 
        ${ixws_BINARY_DIR}/ixwebsocket.lib)
    endif()
else()
    target_link_libraries(openmpRun PRIVATE 
    ${ixws_BINARY_DIR}/libixwebsocket.a)
    
endif()
target_link_libraries(openmpRun PRIVATE ZLIB::ZLIB)
target_include_directories(openmpRun PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)

target_link_libraries(openmpRun PUBLIC OpenMP::OpenMP_CXX)
