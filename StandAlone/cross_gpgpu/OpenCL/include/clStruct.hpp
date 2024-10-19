#pragma once

#include <CL/opencl.hpp>
#include "RunnerInterface.hpp"
#include "CL_Wrapper.h"
#include "cl_global_custom.h"
#include "okl_embed.hpp"

struct Genv{
    std::vector<Platform> PF;
    Device DV;
    Context CT;
};


struct Gcodes{
    Kernel EXP6STFT;
    Kernel EXP7STFT;
    Kernel EXP8STFT;
    Kernel EXP9STFT;
    Kernel EXP10STFT;
    Kernel EXP11STFT;
    
    Kernel EXPCommon;
    Kernel Overlap;

    Kernel DCRemove;
    Kernel Hanning;
    Kernel Hamming;
    Kernel Blackman;
    Kernel Nuttall;
    Kernel Blackman_Nuttall;
    Kernel Blackman_Harris;
    Kernel FlatTop;
    Kernel Gaussian;
    Kernel HalfComplex;
    Kernel toPower;
};

struct clData{
    Genv* env;
    Gcodes* kens;
    CommandQueue* cq;
};