#pragma once
#include <string>

struct ExePath{
    std::string CUDAexe = "./cudaRun";
    std::string OpenCLExe = "./openclRun";
    std::string OpenMPExe = "./openmpRun";
    std::string SerialExe = "./serialRun";
};
