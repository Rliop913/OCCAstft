#!/bin/bash




occa translate -m opencl ./include/kernel.okl > ./cross_gpgpu/OpenCL/kernel/compiled_code.cl
occa translate -m cuda ./include/kernel.okl > ./cross_gpgpu/CUDA/kernel/compiled_coda.hpp
occa translate -m openmp ./include/kernel.okl > ./cross_gpgpu/OpenMP/kernel/compiled_code.hpp


python KERNEL_Embedder.py