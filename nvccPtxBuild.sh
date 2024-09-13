#!/bin/bash


cudaVersion=$(nvcc --version | grep release | cut -d " " -f 5 | sed "s/\./_/" | sed "s/,//")

nvcc -arch=compute_52 -code=compute_52 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_52_${cudaVersion}.ptx
nvcc -arch=compute_61 -code=compute_61 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_61_${cudaVersion}.ptx
nvcc -arch=compute_70 -code=compute_70 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_70_${cudaVersion}.ptx
nvcc -arch=compute_75 -code=compute_75 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_75_${cudaVersion}.ptx
nvcc -arch=compute_80 -code=compute_80 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_80_${cudaVersion}.ptx
nvcc -arch=compute_90 -code=compute_90 -ptx ./cross_gpgpu/CUDA/kernel/radixALL.cu -o ./cross_gpgpu/CUDA/kernel/radixALL_90_${cudaVersion}.ptx

printf "#pragma once\nclass okl_embed52 {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_52_${cudaVersion}.ptx > ./cross_gpgpu/CUDA/kernel/temp_52_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed61 {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_61_${cudaVersion}.ptx > ./cross_gpgpu/CUDA/kernel/temp_61_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed70 {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_70_${cudaVersion}.ptx > ./cross_gpgpu/CUDA/kernel/temp_70_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed75 {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_75_${cudaVersion}.ptx > ./cross_gpgpu/CUDA/kernel/temp_75_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed80 {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_80_${cudaVersion}.ptx > ./cross_gpgpu/CUDA/kernel/temp_80_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed90 {\n public:\n const char* ptx_code = \n R\"(" | cat - ./cross_gpgpu/CUDA/kernel/radixALL_90_${cudaVersion}.ptx > ./cross_gpgpu/CUDA/kernel/temp_90_${cudaVersion}.txt




{
    cat ./cross_gpgpu/CUDA/kernel/temp_52_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_52_${cudaVersion}.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_61_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_61_${cudaVersion}.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_70_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_70_${cudaVersion}.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_75_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_75_${cudaVersion}.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_80_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_80_${cudaVersion}.hpp
{
    cat ./cross_gpgpu/CUDA/kernel/temp_90_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./cross_gpgpu/CUDA/kernel/okl_embed_90_${cudaVersion}.hpp
