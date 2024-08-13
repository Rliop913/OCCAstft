#!/bin/bash




occa translate -m opencl ./include/kernel.okl > ./cross_gpgpu/OpenCL/kernel/compiled_code.cl
occa translate -m cuda ./include/kernel.okl > ./cross_gpgpu/CUDA/kernel/compiled_cuda.cu
occa translate -m openmp ./include/kernel.okl > ./cross_gpgpu/OpenMP/kernel/compiled_opemmp.hpp
occa translate -D __SERIAL -m serial ./include/kernel.okl > ./cross_gpgpu/Serial/kernel/compiled_serial.hpp


python KERNEL_Embedder.py
