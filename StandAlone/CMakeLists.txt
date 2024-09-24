cmake_minimum_required(VERSION 3.5.0)
#set(CMAKE_C_COMPILER clang)
#set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(occaSTFT VERSION 1.0.0 LANGUAGES C CXX)

include(FetchContent)
if(WIN32)#window
add_definitions(-DOS_WINDOWS)
add_definitions(-DWIN32_LEAN_AND_MEAN)
else()
add_definitions(-DOS_POSIX)

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
find_package(CapnProto)
if(NOT CapnProto_FOUND)
    
    message(STATUS "CapnProto not found, running capnpsetter.sh...")
    execute_process(
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/capnpsetter.sh)
    find_package(CapnProto REQUIRED)
endif()
if(NOT WIN32)
    find_package(clFFT)
    if(NOT clFFT_FOUND)
        execute_process(
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/clfftsetter.sh
        )
        
        find_package(clFFT REQUIRED)
    endif()
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

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

if(clFFT_FOUND AND CUDA_FOUND)
add_subdirectory(${CMAKE_SOURCE_DIR}/cross_gpgpu/testRunner)
add_subdirectory(${CMAKE_SOURCE_DIR}/cross_gpgpu/occaprofileRunner)
endif()


set(mainsrc 
    ${CMAKE_SOURCE_DIR}/src/runtimeChecker.cpp
    ${CMAKE_SOURCE_DIR}/src/STFTProxy.cpp
    ${CMAKE_SOURCE_DIR}/src/FFTStruct.cpp
    ${CMAKE_SOURCE_DIR}/src/FFTcapnp.capnp.c++
    ${CMAKE_SOURCE_DIR}/src/FFTStruct_SetGet.cpp
    )

set(
    testersrc
    ${CMAKE_SOURCE_DIR}/src/runtimeChecker.cpp
    ${CMAKE_SOURCE_DIR}/src/STFTProxy.cpp
    ${CMAKE_SOURCE_DIR}/src/FFTStruct.cpp
    ${CMAKE_SOURCE_DIR}/src/FFTcapnp.capnp.c++
    ${CMAKE_SOURCE_DIR}/src/FFTStruct_SetGet.cpp
    ${CMAKE_SOURCE_DIR}/Profile_Code/profileRunners.cpp
    
)

add_library(STFTMain ${mainsrc})
add_executable(TESTER ${testersrc})

target_link_libraries(STFTMain PRIVATE CapnProto::capnp CapnProto::kj)
target_link_libraries(TESTER PRIVATE CapnProto::capnp CapnProto::kj)



if(WIN32)
target_link_libraries(STFTMain PRIVATE ws2_32)
target_link_libraries(TESTER PRIVATE ws2_32)

    if(MSVC)
        target_link_libraries(STFTMain PRIVATE 
            $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
            $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
            )
        target_link_libraries(TESTER PRIVATE 
        $<$<CONFIG:Release>:${ixws_BINARY_DIR}/Release/ixwebsocket.lib>
        $<$<CONFIG:Debug>:${ixws_BINARY_DIR}/Debug/ixwebsocket.lib>
        )
    else()
        target_link_libraries(STFTMain PRIVATE ${ixws_BINARY_DIR}/ixwebsocket.lib)
        target_link_libraries(TESTER PRIVATE ${ixws_BINARY_DIR}/ixwebsocket.lib)
    endif()
    target_link_libraries(STFTMain PRIVATE ws2_32)
    target_link_libraries(TESTER PRIVATE ws2_32)
else()
    target_link_libraries(STFTMain PRIVATE ${ixws_BINARY_DIR}/libixwebsocket.a)
    target_link_libraries(TESTER PRIVATE ${ixws_BINARY_DIR}/libixwebsocket.a)
endif()

target_link_libraries(STFTMain PRIVATE ZLIB::ZLIB)
target_link_libraries(TESTER PRIVATE ZLIB::ZLIB)

target_include_directories(STFTMain PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)
target_include_directories(TESTER PRIVATE ${ixws_SOURCE_DIR}/ixwebsocket)

# find_package(OCCA REQUIRED)
# add_executable(OCCATEST 
# ${CMAKE_SOURCE_DIR}/src/occa.cpp
# ${CMAKE_SOURCE_DIR}/miniaudio/miniaudio.c
# )
# target_include_directories(OCCATEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(OCCATEST PRIVATE ${CMAKE_SOURCE_DIR}/miniaudio)

# target_link_libraries(OCCATEST PRIVATE OCCA::libocca)