
#include <CL/opencl.hpp>
#include "RunnerInterface.hpp"
#include "CL_Wrapper.h"
#include "cl_global_custom.h"
#include "okl_embedded.h"
struct Genv{
    std::vector<Platform> PF;
    Device DV;
    Context CT;
    CommandQueue CQ;
};


struct Gcodes{
    Kernel R10STFT;
    Kernel R11STFT;
    Kernel R12STFT;
    Kernel R13STFT;
    Kernel R14STFT;
    Kernel R15STFT;
    Kernel toPower;
};

void
Runner::InitEnv()
{
    env = new Genv;
    env->PF = clboost::get_platform();
    env->DV = clboost::get_gpu_device(env->PF);
    env->CT = clboost::get_context(env->DV);
    env->CQ = clboost::make_cq(env->CT, env->DV);
}

void
Runner::UnInit()
{
    //nothing
}

void
Runner::BuildKernel()
{
    kens = new Gcodes;
    okl_embed clCodes;
    Program codeBase = clboost::make_prog(clCodes.compiled, env->CT, env->DV);
    kens->R10STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW10_STH_STFT_0");
    kens->R11STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW11_STH_STFT_0");
    kens->R12STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW12_STH_STFT_0");
    kens->R13STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW13_STH_STFT_0");
    kens->R14STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW14_STH_STFT_0");
    kens->R15STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW15_STH_STFT_0");
    
    kens->toPower = clboost::make_kernel(codeBase, "_occa_toPower_0");
    

}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio)
{
    const unsigned int FullSize = inData.size();
    const int windowSize = 1 << windowRadix;
    const int qtConst = toQuot(FullSize, overlapRatio, windowSize);
    const unsigned int OFullSize = qtConst * windowSize;
    const unsigned int OHalfSize = OFullSize / 2;
    const unsigned int OMove     = windowSize * (1.0f - overlapRatio);
    Buffer inMem = clboost::HTDCopy<float>(env->CT, FullSize, inData.data());
    Buffer outMem = clboost::DMEM<float>(env->CT, OHalfSize);
    Buffer tempMem = clboost::DMEM<cplx_t>(env->CT, OFullSize);
    
    std::vector<int> error_container(30 + windowRadix);
    int error_itr = 0;
    Kernel* RAIO;
    int workGroupSize = 0;
    switch (windowRadix)
    {
    case 15:
        RAIO = &(kens->R15STFT);
        workGroupSize = 1024;
        break;
    case 14:
        RAIO = &(kens->R14STFT);
        workGroupSize = 1024;
        break;
    case 13:
        RAIO = &(kens->R13STFT);
        workGroupSize = 1024;
        break;
    case 12:
        RAIO = &(kens->R12STFT);
        workGroupSize = 1024;
        break;
    case 11:
        RAIO = &(kens->R11STFT);
        workGroupSize = 1024;
        break;
    case 10:
        RAIO = &(kens->R10STFT);
        workGroupSize = 512;
        break;
    default:
        break;
    }
    
    error_container[error_itr++] = RAIO->setArg(0, inMem);
    error_container[error_itr++] = RAIO->setArg(1, qtConst);
    error_container[error_itr++] = RAIO->setArg(2, FullSize);
    error_container[error_itr++] = RAIO->setArg(3, OMove);
    error_container[error_itr++] = RAIO->setArg(4, OHalfSize);
    error_container[error_itr++] = RAIO->setArg(5, tempMem);
    
    error_container[error_itr++] = kens->toPower.setArg(0, tempMem);
    error_container[error_itr++] = kens->toPower.setArg(1, outMem);
    error_container[error_itr++] = kens->toPower.setArg(2, OHalfSize);
    error_container[error_itr++] = kens->toPower.setArg(3, windowRadix);
    
    error_container[error_itr++] = clboost::enq_q(env->CQ, (*RAIO), OFullSize, workGroupSize);
    
    error_container[error_itr++] = clboost::enq_q(env->CQ, kens->toPower, OHalfSize, LOCAL_SIZE);
    std::vector<float> outData(OHalfSize);
    error_container[error_itr++] = clboost::q_read(env->CQ, outMem, true, OHalfSize, outData.data());

    for(auto Eitr : error_container){
        if(Eitr != CL_SUCCESS){
            return std::nullopt;
        }
    }

    return std::move(outData);
}