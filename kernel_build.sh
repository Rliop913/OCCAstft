#!/bin/bash




occa translate -D ROOTISBASH -m opencl ./include/Radix10.okl \
> ./cross_gpgpu/OpenCL/kernel/radix10.cl

occa translate -D ROOTISBASH -m opencl ./include/Radix11.okl \
> ./cross_gpgpu/OpenCL/kernel/radix11.cl

occa translate -D ROOTISBASH -m opencl ./include/Radix12.okl \
> ./cross_gpgpu/OpenCL/kernel/radix12.cl

occa translate -D ROOTISBASH -m opencl ./include/Radix13.okl \
> ./cross_gpgpu/OpenCL/kernel/radix13.cl

occa translate -D ROOTISBASH -m opencl ./include/Radix14.okl \
> ./cross_gpgpu/OpenCL/kernel/radix14.cl

occa translate -D ROOTISBASH -m opencl ./include/Radix15.okl \
> ./cross_gpgpu/OpenCL/kernel/radix15.cl

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./include/RadixCommon.okl
}> ./cross_gpgpu/CUDA/kernel/radixCommon.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./include/Radix10.okl
}> ./cross_gpgpu/CUDA/kernel/radix10.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./include/Radix11.okl
}> ./cross_gpgpu/CUDA/kernel/radix11.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./include/Radix12.okl
}> ./cross_gpgpu/CUDA/kernel/radix12.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./include/Radix13.okl
}> ./cross_gpgpu/CUDA/kernel/radix13.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./include/Radix14.okl
}> ./cross_gpgpu/CUDA/kernel/radix14.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./include/Radix15.okl
}> ./cross_gpgpu/CUDA/kernel/radix15.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m openmp ./include/RadixALL.okl 
}> ./cross_gpgpu/OpenMP/kernel/compiled.hpp

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m serial ./include/RadixALL.okl 
}> ./cross_gpgpu/Serial/kernel/compiled.hpp

python KERNEL_Embedder.py

nvcc -ptx ./cross_gpgpu/CUDA/kernel/radix10.cu -o ./cross_gpgpu/CUDA/kernel/radix10.ptx
nvcc -ptx ./cross_gpgpu/CUDA/kernel/radix11.cu -o ./cross_gpgpu/CUDA/kernel/radix11.ptx
nvcc -ptx -arch=sm_60 ./cross_gpgpu/CUDA/kernel/radix12.cu -o ./cross_gpgpu/CUDA/kernel/radix12.ptx
nvcc -ptx -arch=sm_60 ./cross_gpgpu/CUDA/kernel/radixCommon.cu -o ./cross_gpgpu/CUDA/kernel/radixCommon.ptx
nvcc -ptx -arch=sm_60 ./cross_gpgpu/CUDA/kernel/radixCommon.cu -o ./build/cross_gpgpu/CUDA/radixCommon.ptx
# nvcc -ptx ./cross_gpgpu/CUDA/kernel/radix13.cu -o ./cross_gpgpu/CUDA/kernel/radix13.ptx
# nvcc -ptx ./cross_gpgpu/CUDA/kernel/radix14.cu -o ./cross_gpgpu/CUDA/kernel/radix14.ptx
# nvcc -ptx ./cross_gpgpu/CUDA/kernel/radix15.cu -o ./cross_gpgpu/CUDA/kernel/radix15.ptx

nvcc -fatbin ^
-gencode arch=compute_52,code=sm_52 ^
-gencode arch=compute_60,code=sm_60 ^
-gencode arch=compute_75,code=sm_75 ^
 ./cross_gpgpu/CUDA/kernel/radix10.cu -o ./build/cross_gpgpu/CUDA/radix10.fatbin

nvcc -fatbin  ^
-gencode arch=compute_52,code=sm_52 ^
-gencode arch=compute_60,code=sm_60 ^
-gencode arch=compute_75,code=sm_75 ^
./cross_gpgpu/CUDA/kernel/radix11.cu -o ./build/cross_gpgpu/CUDA/radix11.fatbin

nvcc -fatbin  ^
-gencode arch=compute_60,code=sm_60 ^
-gencode arch=compute_75,code=sm_75 ^
./cross_gpgpu/CUDA/kernel/radix12.cu -o ./build/cross_gpgpu/CUDA/radix12.fatbin

nvcc -fatbin  ^
-gencode arch=compute_52,code=sm_52 ^
-gencode arch=compute_60,code=sm_60 ^
-gencode arch=compute_75,code=sm_75 ^ 
./cross_gpgpu/CUDA/kernel/radix13.cu -o ./build/cross_gpgpu/CUDA/radix13.fatbin
nvcc -fatbin  ^
-gencode arch=compute_52,code=sm_52 ^
-gencode arch=compute_60,code=sm_60 ^
-gencode arch=compute_75,code=sm_75 ^ 
./cross_gpgpu/CUDA/kernel/radix14.cu -o ./build/cross_gpgpu/CUDA/radix14.fatbin
nvcc -fatbin  ^
-gencode arch=compute_52,code=sm_52 ^
-gencode arch=compute_60,code=sm_60 ^
-gencode arch=compute_75,code=sm_75 ^
./cross_gpgpu/CUDA/kernel/radix15.cu -o ./build/cross_gpgpu/CUDA/radix15.fatbin