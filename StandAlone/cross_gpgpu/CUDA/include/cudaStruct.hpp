#pragma once

#include "RunnerInterface.hpp"

#include "okl_embed_52_11_6.hpp"
#include "okl_embed_52_12_1.hpp"
#include "okl_embed_52_12_3.hpp"

#include "okl_embed_61_11_6.hpp"
#include "okl_embed_61_12_1.hpp"
#include "okl_embed_61_12_3.hpp"

#include "okl_embed_70_11_6.hpp"
#include "okl_embed_70_12_1.hpp"
#include "okl_embed_70_12_3.hpp"

#include "okl_embed_75_11_6.hpp"
#include "okl_embed_75_12_1.hpp"
#include "okl_embed_75_12_3.hpp"

#include "okl_embed_80_11_6.hpp"
#include "okl_embed_80_12_1.hpp"
#include "okl_embed_80_12_3.hpp"


#include "okl_embed_90_12_1.hpp"
#include "okl_embed_90_12_3.hpp"

#include <cuda.h>


#define LOAD_PTX(buildName, ValueName, IF_Fail_DO)\
    buildName ValueName;\
    if(cuModuleLoadData(&(env->EXPAll), ValueName.ptx_code) != CUDA_SUCCESS)\
    {\
        IF_Fail_DO;\
    }

// Genv: Structure to hold the GPGPU environment settings and resources.
struct Genv{
    CUdevice device;
    CUcontext context;
    CUmodule EXPAll;
    
    
    //
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    CUfunction EXP6STFT;
    CUfunction EXP7STFT;
    CUfunction EXP8STFT;
    CUfunction EXP9STFT;
    CUfunction EXP10STFT;
    CUfunction EXP11STFT;
    
    CUfunction EXPCommon;
    CUfunction Overlap;
    CUfunction DCRemove;

    CUfunction Hanning;
    CUfunction Hamming;
    CUfunction Blackman;
    CUfunction Nuttall;
    CUfunction Blackman_Nuttall;
    CUfunction Blackman_Harris;
    CUfunction FlatTop;
    CUfunction Gaussian;

    CUfunction HalfComplex;
    CUfunction toPower;

};

struct cudaData{
    Genv* env;
    Gcodes* kens;
    CUstream* strm;
    unsigned int qtConst;
};