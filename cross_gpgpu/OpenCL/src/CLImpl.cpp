
#include <CL/opencl.hpp>
#include "RunnerInterface.hpp"
#include "CL_Wrapper.h"
#include "cl_global_custom.h"
#include "okl_embedded.h"
struct Genv{
    std::vector<Platform> PF;
    Device DV;
    Context CT;
};


struct Gcodes{
    Kernel R10STFT;
    Kernel R11STFT;
    Kernel RadixCommon;
    Kernel Overlap;
    Kernel DCRemove;
    Kernel Windowing;
    Kernel toPower;
};

void
Runner::InitEnv()
{
    env = new Genv;
    env->PF = clboost::get_platform();
    env->DV = clboost::get_gpu_device(env->PF);
    env->CT = clboost::get_context(env->DV);
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
    Program codeBase = clboost::make_prog(clCodes.radixALL, env->CT, env->DV);
    kens->R10STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW10_STH_STFT_0");
    kens->R11STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW11_STH_STFT_0");
    kens->RadixCommon = clboost::make_kernel(codeBase, "_occa_StockHamDITCommon_0");
    kens->Overlap = clboost::make_kernel(codeBase, "_occa_Overlap_Common_0");
    kens->DCRemove = clboost::make_kernel(codeBase, "_occa_DCRemove_Common_0");
    kens->Windowing = clboost::make_kernel(codeBase, "_occa_Window_Common_0");
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

    CommandQueue CQ = clboost::make_cq(env->CT, env->DV);
    Buffer inMem = clboost::HTDCopy<float>(env->CT, FullSize, inData.data());
    Buffer RealMem = clboost::DMEM<float>(env->CT, OFullSize);
    Buffer ImagMem = clboost::DMEM<float>(env->CT, OFullSize);
    Buffer outMem = Buffer(env->CT, CL_MEM_WRITE_ONLY, sizeof(float)* OFullSize);
    
    std::vector<int> error_container;
    
    switch (windowRadix)
    {
    case 11:
        error_container.push_back(kens->R11STFT.setArg(0, inMem));
        error_container.push_back(kens->R11STFT.setArg(1, qtConst));
        error_container.push_back(kens->R11STFT.setArg(2, FullSize));
        error_container.push_back(kens->R11STFT.setArg(3, OMove));
        error_container.push_back(kens->R11STFT.setArg(4, OHalfSize));
        error_container.push_back(kens->R11STFT.setArg(5, RealMem));
        error_container.push_back(kens->R11STFT.setArg(6, ImagMem));
        error_container.push_back(clboost::enq_q(CQ, kens->R11STFT, OHalfSize, 1024));

        error_container.push_back(kens->toPower.setArg(1, RealMem));
        error_container.push_back(kens->toPower.setArg(2, ImagMem));
        break;
    case 10:
        error_container.push_back(kens->R10STFT.setArg(0, inMem));
        error_container.push_back(kens->R10STFT.setArg(1, qtConst));
        error_container.push_back(kens->R10STFT.setArg(2, FullSize));
        error_container.push_back(kens->R10STFT.setArg(3, OMove));
        error_container.push_back(kens->R10STFT.setArg(4, OHalfSize));
        error_container.push_back(kens->R10STFT.setArg(5, RealMem));
        error_container.push_back(kens->R10STFT.setArg(6, ImagMem));
        error_container.push_back(clboost::enq_q(CQ, kens->R10STFT, OHalfSize, 512));

        error_container.push_back(kens->toPower.setArg(1, RealMem));
        error_container.push_back(kens->toPower.setArg(2, ImagMem));
        
        break;
    default:
        error_container.push_back(kens->Overlap.setArg(0, inMem));
        error_container.push_back(kens->Overlap.setArg(1, OFullSize));
        error_container.push_back(kens->Overlap.setArg(2, FullSize));
        error_container.push_back(kens->Overlap.setArg(3, windowRadix));
        error_container.push_back(kens->Overlap.setArg(4, OMove));
        error_container.push_back(kens->Overlap.setArg(5, RealMem));
        
        error_container.push_back(clboost::enq_q(CQ, kens->Overlap, OFullSize, 1024));

        error_container.push_back(kens->DCRemove.setArg(0, RealMem));
        error_container.push_back(kens->DCRemove.setArg(1, OFullSize));
        error_container.push_back(kens->DCRemove.setArg(2, windowSize));
          
        error_container.push_back(clboost::enq_q(CQ, kens->DCRemove, qtConst * 64, 64));
  
        error_container.push_back(kens->Windowing.setArg(0, RealMem));
        error_container.push_back(kens->Windowing.setArg(1, OFullSize));
        error_container.push_back(kens->Windowing.setArg(2, windowRadix));
          
        error_container.push_back(clboost::enq_q(CQ, kens->Windowing, OFullSize, 1024));
        Buffer ORealMem = clboost::DMEM<float>(env->CT, OFullSize);
        Buffer OImagMem = clboost::DMEM<float>(env->CT, OFullSize);
        unsigned int HwindowSize = windowSize >> 1;
        error_container.push_back(kens->RadixCommon.setArg(4, HwindowSize));
          
        error_container.push_back(kens->RadixCommon.setArg(6, OHalfSize));
        error_container.push_back(kens->RadixCommon.setArg(7, windowRadix));
        
        for(unsigned int stage = 0; stage < windowRadix; ++stage)
        {
            error_container.push_back(kens->RadixCommon.setArg(5, stage));
            if(stage % 2 == 0)
            {
                error_container.push_back(kens->RadixCommon.setArg(0, RealMem));
                error_container.push_back(kens->RadixCommon.setArg(1, ImagMem));
                error_container.push_back(kens->RadixCommon.setArg(2, ORealMem));
                error_container.push_back(kens->RadixCommon.setArg(3, OImagMem));
            }
            else
            {
                error_container.push_back(kens->RadixCommon.setArg(0, ORealMem));
                error_container.push_back(kens->RadixCommon.setArg(1, OImagMem));
                error_container.push_back(kens->RadixCommon.setArg(2, RealMem));
                error_container.push_back(kens->RadixCommon.setArg(3, ImagMem));
            }
            error_container.push_back(clboost::enq_q(CQ, kens->RadixCommon, OHalfSize, 256));
        }
        if(windowRadix % 2 == 0)
        {
            error_container.push_back(kens->toPower.setArg(1, RealMem));
            error_container.push_back(kens->toPower.setArg(2, ImagMem));
        }
        else
        {
            error_container.push_back(kens->toPower.setArg(1, ORealMem));
            error_container.push_back(kens->toPower.setArg(2, OImagMem));
        }
        break;
    }

    
    error_container.push_back(kens->toPower.setArg(0, outMem));
    error_container.push_back(kens->toPower.setArg(3, OFullSize));
    error_container.push_back(kens->toPower.setArg(4, windowRadix));
    
    error_container.push_back(clboost::enq_q(CQ, kens->toPower, OFullSize, 256));
    
    std::vector<float> outData(OFullSize);
    error_container.push_back(clboost::q_read(CQ, outMem, true, OFullSize, outData.data()));
    
    for(auto Eitr : error_container){
        if(Eitr != CL_SUCCESS){
            return std::nullopt;
        }
    }

    return std::move(outData);
}