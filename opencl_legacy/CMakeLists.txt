cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS -std=c++20)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
project("OpenCL_BATCH_STFT")

Include(FetchContent)
FetchContent_Declare(
  OpenCL
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
  GIT_TAG        v2024.05.08 
)
#target_include_directories(main PRIVATE ${TCLAP_INCLUDE_DIRS})

set(BUILD_TESTING OFF)
set(BUILD_DOCS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_TESTS OFF)
set(OPENCL_SDK_TEST_SAMPLES OFF)
set(OPENCL_SDK_BUILD_UTILITY_LIBRARIES OFF)
set(OPENCL_SDK_BUILD_SAMPLES OFF)
set(OPENCL_SDK_BUILD_OPENGL_SAMPLES OFF)

add_definitions(-DBUILD_TESTING=OFF)


# set(TCLAP_INCLUDE_PATH ${TCLAP_SOURCE_DIR}/include/tclap)
# message(${CMAKE_CURRENT_BINARY_DIR}/_deps/stb-external-src)
# set(Stb_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR}/_deps/stb-external-src)
FetchContent_MakeAvailable(OpenCL)
#add_subdirectory(${opencl_SOURCE_DIR})
#enable_testing()
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_CXX_STANDARD 20)

#add_subdirectory(${CMAKE_SOURCE_DIR}/soundtouch-master/soundtouch)




set(main_src
${CMAKE_SOURCE_DIR}/src/STFT.cpp
${CMAKE_SOURCE_DIR}/include/CL_Wrapper/src/cl_FACADE.cpp
${CMAKE_SOURCE_DIR}/include/CL_Wrapper/src/cl_global_custom.cpp
)

set(test_src
${CMAKE_SOURCE_DIR}/catch_tests/test.cpp
)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/CL_Wrapper/include
  ${CMAKE_SOURCE_DIR}/include/kernels
  )
  
  add_library(${PROJECT_NAME} STATIC ${main_src})
  #add_executable(test_module ${main_src} ${test_src})
  add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=300)
  
  target_link_libraries(${PROJECT_NAME} PUBLIC
  OpenCL::OpenCL
  )
  # target_link_libraries(test_module PUBLIC
  # OpenCL::OpenCL
  # )
  set(CMAKE_BUILD_TYPE Release)

  
  
  if(WIN32)
	
  elseif(UNIX)
	add_definitions(-DFOR_LINUX_BUILD)
  endif()
  
  
  # target_link_libraries(test_module PUBLIC Catch2::Catch2WithMain)
  # set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
  # set_property(TARGET test_module PROPERTY CXX_STANDARD 20)
  
  # # target_include_directories(
    # 	${PROJECT_NAME} PRIVATE
    
    # )
    # target_include_directories(
      # 	test_module PRIVATE
      #     ${CMAKE_SOURCE_DIR}/STFT_SYCL/include
      #     ${CMAKE_SOURCE_DIR}/STFT_SYCL/GPGPU_frame/include
      #     ${CMAKE_SOURCE_DIR}/STFT_SYCL/GPGPU_frame/OpenCL-SDK/install/include
      # )
      set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
      
      #${CMAKE_SOURCE_DIR}/soundtouch_include/
      #include(CTest)
      #include(Catch)
      #catch_discover_tests(test_module)
      add_compile_options(-O3)
      