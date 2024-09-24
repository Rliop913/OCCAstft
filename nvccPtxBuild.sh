#!/bin/bash


cudaVersion=$(nvcc --version | grep release | cut -d " " -f 5 | sed "s/\./_/" | sed "s/,//")

nvcc -arch=compute_52 -code=compute_52 -ptx ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN.cu -o ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_52_${cudaVersion}.ptx
nvcc -arch=compute_61 -code=compute_61 -ptx ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN.cu -o ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_61_${cudaVersion}.ptx
nvcc -arch=compute_70 -code=compute_70 -ptx ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN.cu -o ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_70_${cudaVersion}.ptx
nvcc -arch=compute_75 -code=compute_75 -ptx ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN.cu -o ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_75_${cudaVersion}.ptx
nvcc -arch=compute_80 -code=compute_80 -ptx ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN.cu -o ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_80_${cudaVersion}.ptx
nvcc -arch=compute_90 -code=compute_90 -ptx ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN.cu -o ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_90_${cudaVersion}.ptx

printf "#pragma once\nclass okl_embed52_${cudaVersion} {\n public:\n const char* ptx_code = \n R\"(" | cat - ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_52_${cudaVersion}.ptx > ./StandAlone/cross_gpgpu/CUDA/kernel/temp_52_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed61_${cudaVersion} {\n public:\n const char* ptx_code = \n R\"(" | cat - ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_61_${cudaVersion}.ptx > ./StandAlone/cross_gpgpu/CUDA/kernel/temp_61_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed70_${cudaVersion} {\n public:\n const char* ptx_code = \n R\"(" | cat - ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_70_${cudaVersion}.ptx > ./StandAlone/cross_gpgpu/CUDA/kernel/temp_70_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed75_${cudaVersion} {\n public:\n const char* ptx_code = \n R\"(" | cat - ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_75_${cudaVersion}.ptx > ./StandAlone/cross_gpgpu/CUDA/kernel/temp_75_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed80_${cudaVersion} {\n public:\n const char* ptx_code = \n R\"(" | cat - ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_80_${cudaVersion}.ptx > ./StandAlone/cross_gpgpu/CUDA/kernel/temp_80_${cudaVersion}.txt
printf "#pragma once\nclass okl_embed90_${cudaVersion} {\n public:\n const char* ptx_code = \n R\"(" | cat - ./StandAlone/cross_gpgpu/CUDA/kernel/STFT_MAIN_90_${cudaVersion}.ptx > ./StandAlone/cross_gpgpu/CUDA/kernel/temp_90_${cudaVersion}.txt




{
    cat ./StandAlone/cross_gpgpu/CUDA/kernel/temp_52_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./StandAlone/cross_gpgpu/CUDA/kernel/okl_embed_52_${cudaVersion}.hpp
{
    cat ./StandAlone/cross_gpgpu/CUDA/kernel/temp_61_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./StandAlone/cross_gpgpu/CUDA/kernel/okl_embed_61_${cudaVersion}.hpp
{
    cat ./StandAlone/cross_gpgpu/CUDA/kernel/temp_70_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./StandAlone/cross_gpgpu/CUDA/kernel/okl_embed_70_${cudaVersion}.hpp
{
    cat ./StandAlone/cross_gpgpu/CUDA/kernel/temp_75_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./StandAlone/cross_gpgpu/CUDA/kernel/okl_embed_75_${cudaVersion}.hpp
{
    cat ./StandAlone/cross_gpgpu/CUDA/kernel/temp_80_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./StandAlone/cross_gpgpu/CUDA/kernel/okl_embed_80_${cudaVersion}.hpp
{
    cat ./StandAlone/cross_gpgpu/CUDA/kernel/temp_90_${cudaVersion}.txt - <<EOF
    )";};
EOF
}> ./StandAlone/cross_gpgpu/CUDA/kernel/okl_embed_90_${cudaVersion}.hpp
