#!/bin/bash


occa translate -D ROOTISBASH -m opencl ./include/RadixALL.okl \
> ./cross_gpgpu/OpenCL/kernel/radixALL.cl

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m cuda ./include/RadixALL.okl
}> ./cross_gpgpu/CUDA/kernel/radixALL.cu

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m openmp ./include/RadixALL.okl 
}> ./cross_gpgpu/OpenMP/kernel/compiled.hpp

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m serial ./include/RadixALL.okl 
}> ./cross_gpgpu/Serial/kernel/compiled.hpp

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m hip ./include/RadixALL.okl 
}> ./cross_gpgpu/HIP/kernel/compiled.hpp

{
    echo "#define _USE_MATH_DEFINES";
    occa translate -D __NEED_PI -D ROOTISBASH -m metal ./include/RadixALL.okl 
}> ./cross_gpgpu/METAL/kernel/compiled.hpp


nvcc -arch=compute_52 -code=compute_52 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_52.ptx
nvcc -arch=compute_61 -code=compute_61 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_61.ptx
nvcc -arch=compute_70 -code=compute_70 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_70.ptx
nvcc -arch=compute_75 -code=compute_75 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_75.ptx
nvcc -arch=compute_80 -code=compute_80 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_80.ptx
nvcc -arch=compute_90 -code=compute_90 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_90.ptx

python CL_Embedder.py


printf "#pragma once\nclass okl_embed {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL.ptx > ./cross_gpgpu/CUDA/kernel/temp.txt

printf "#pragma once\nclass okl_embed {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_52.ptx > ./cross_gpgpu/CUDA/kernel/temp_52.txt
printf "#pragma once\nclass okl_embed {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_61.ptx > ./cross_gpgpu/CUDA/kernel/temp_61.txt
printf "#pragma once\nclass okl_embed {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_70.ptx > ./cross_gpgpu/CUDA/kernel/temp_70.txt
printf "#pragma once\nclass okl_embed {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_75.ptx > ./cross_gpgpu/CUDA/kernel/temp_75.txt
printf "#pragma once\nclass okl_embed {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_80.ptx > ./cross_gpgpu/CUDA/kernel/temp_80.txt
printf "#pragma once\nclass okl_embed {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_90.ptx > ./cross_gpgpu/CUDA/kernel/temp_90.txt





{
    cat ./cross_gpgpu/CUDA/kernel/temp.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed.hpp

{
    cat ./cross_gpgpu/CUDA/kernel/temp_52.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_52.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_61.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_61.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_70.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_70.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_75.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_75.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_80.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_80.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_90.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_90.hpp
