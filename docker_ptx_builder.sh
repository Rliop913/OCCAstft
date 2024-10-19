#!/bin/bash
cudaVersions=(nvidia/cuda:11.0.3-devel-ubuntu20.04 \
nvidia/cuda:11.3.1-devel-ubi8 \
nvidia/cuda:11.5.2-devel-ubi8 \
nvidia/cuda:11.7.1-devel-ubi8 \
nvidia/cuda:12.0.0-devel-ubi8 \
nvidia/cuda:12.1.0-devel-ubi8 \
nvidia/cuda:12.2.2-devel-ubi8 \
nvidia/cuda:12.3.2-devel-ubi8 \
nvidia/cuda:12.4.0-devel-ubi8 \
nvidia/cuda:12.5.0-devel-ubi8 \
nvidia/cuda:12.6.2-devel-ubuntu22.04)

printf "#pragma once\n" > ./StandAlone/cross_gpgpu/CUDA/kernel/include_kernels.hpp

for CUversion in "${cudaVersions[@]}"; do
    sudo docker pull "$CUversion"
    sudo docker run -v ./:/home/ "$CUversion" bash -c "home/position_setter_for_docker.sh"
done

