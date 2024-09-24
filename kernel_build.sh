#!/bin/bash


occa translate -D ROOTISBASH -m opencl ./OKL/STFT_MAIN.okl \
> ./StandAlone/cross_gpgpu/OpenCL/kernel/STFT_MAIN.cl

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./OKL/STFT_MAIN.okl
}> ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m openmp ./OKL/STFT_MAIN.okl 
}> ./StandAlone/cross_gpgpu/OpenMP/kernel/compiled.hpp

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m serial ./OKL/STFT_MAIN.okl 
}> ./StandAlone/cross_gpgpu/Serial/kernel/compiled.hpp

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m hip ./OKL/STFT_MAIN.okl 
}> ./StandAlone/cross_gpgpu/HIP/kernel/compiled.hpp

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m metal ./OKL/STFT_MAIN.okl 
}> ./StandAlone/cross_gpgpu/METAL/kernel/compiled.hpp

printf "#pragma once\nclass okl_embed {\n public:\n const char* opencl_code = \n R\"(" | cat - ./StandAlone/cross_gpgpu/OpenCL/kernel/STFT_MAIN.cl > ./StandAlone/cross_gpgpu/OpenCL/kernel/temp.txt


{
    cat ./StandAlone/cross_gpgpu/OpenCL/kernel/temp.txt - <<EOF
    )";};
EOF
}> ./StandAlone/cross_gpgpu/OpenCL/kernel/okl_embed.hpp