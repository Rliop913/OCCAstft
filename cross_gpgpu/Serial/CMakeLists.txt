cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(serialSTFT VERSION 0.1.0 LANGUAGES C CXX)




include(FetchContent)


FetchContent_Declare(
  ixws
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  GIT_TAG v11.4.5
)

FetchContent_MakeAvailable(ixws)

find_package(ZLIB REQUIRED)
find_package(CapnProto REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(serialsrc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SerialImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL/RunnerCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTStruct_SetGet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/FFTcapnp.capnp.c++
    
    )



add_executable(serialRun ${serialsrc})

target_link_libraries(serialRun PRIVATE CapnProto::capnp CapnProto::kj)

target_include_directories(serialRun PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../GLOBAL#reverseOOP
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(WIN32)
    if(MSVC)
        target_link_libraries(serialRun PRIVATE
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
    else()
        target_link_libraries(serialRun PRIVATE 
        ${ixws_BINARY_DIR}/ixwebsocket.lib)
    endif()
else()
    target_link_libraries(serialRun PRIVATE 
    ${ixws_BINARY_DIR}/libixwebsocket.a)
    
endif()
target_link_libraries(serialRun PRIVATE ZLIB::ZLIB)
target_include_directories(serialRun PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)