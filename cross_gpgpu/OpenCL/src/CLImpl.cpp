
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
    Kernel R6STFT;
    Kernel R7STFT;
    Kernel R8STFT;
    Kernel R9STFT;
    Kernel R10STFT;
    Kernel R11STFT;
    
    Kernel RadixCommon;
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
    
    kens->R6STFT = clboost::make_kernel(codeBase, "_occa_Stockhpotimized6_0");
    kens->R7STFT = clboost::make_kernel(codeBase, "_occa_Stockhpotimized7_0");
    kens->R8STFT = clboost::make_kernel(codeBase, "_occa_Stockhpotimized8_0");
    kens->R9STFT = clboost::make_kernel(codeBase, "_occa_Stockhpotimized9_0");
    kens->R10STFT = clboost::make_kernel(codeBase, "_occa_Stockhpotimized10_0");
    kens->R11STFT = clboost::make_kernel(codeBase, "_occa_Stockhpotimized11_0");
    
    kens->RadixCommon = clboost::make_kernel(codeBase, "_occa_StockHamDITCommon_0");
    kens->Overlap = clboost::make_kernel(codeBase, "_occa_Overlap_Common_0");
    kens->DCRemove = clboost::make_kernel(codeBase, "_occa_DCRemove_Common_0");
    
    kens->Hanning = clboost::make_kernel(codeBase, "_occa_Window_Hanning_0");
    kens->Hamming = clboost::make_kernel(codeBase, "_occa_Window_Hamming_0");
    kens->Blackman = clboost::make_kernel(codeBase, "_occa_Window_Blackman_0");
    kens->Blackman_Harris = clboost::make_kernel(codeBase, "_occa_Window_Blackman_harris_0");
    kens->Blackman_Nuttall = clboost::make_kernel(codeBase, "_occa_Window_Blackman_Nuttall_0");
    kens->Nuttall = clboost::make_kernel(codeBase, "_occa_Window_Nuttall_0");
    kens->FlatTop = clboost::make_kernel(codeBase, "_occa_Window_FlatTop_0");
    kens->Gaussian = clboost::make_kernel(codeBase, "_occa_Window_Gaussian_0");

    kens->HalfComplex = clboost::make_kernel(codeBase, "_occa_toHalfComplexFormat_0");
    
    kens->toPower = clboost::make_kernel(codeBase, "_occa_toPower_0");

    vps.Hanning = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        kens->Hanning.setArg(0, *((Buffer*)outReal));
        kens->Hanning.setArg(1, OFullSize);
        kens->Hanning.setArg(2, windowSize);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->Hanning, OFullSize, 64);
    };

    vps.Hamming = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        kens->Hamming.setArg(0, *((Buffer*)outReal));
        kens->Hamming.setArg(1, OFullSize);
        kens->Hamming.setArg(2, windowSize);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->Hamming, OFullSize, 64);
    };
    vps.Blackman = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        kens->Blackman.setArg(0, *((Buffer*)outReal));
        kens->Blackman.setArg(1, OFullSize);
        kens->Blackman.setArg(2, windowSize);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->Blackman, OFullSize, 64);
    };

    vps.Nuttall = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        kens->Nuttall.setArg(0, *((Buffer*)outReal));
        kens->Nuttall.setArg(1, OFullSize);
        kens->Nuttall.setArg(2, windowSize);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->Nuttall, OFullSize, 64);
    };

    vps.Blackman_Nuttall = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        kens->Blackman_Nuttall.setArg(0, *((Buffer*)outReal));
        kens->Blackman_Nuttall.setArg(1, OFullSize);
        kens->Blackman_Nuttall.setArg(2, windowSize);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->Blackman_Nuttall, OFullSize, 64);
    };

    vps.Blackman_Harris = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        kens->Blackman_Harris.setArg(0, *((Buffer*)outReal));
        kens->Blackman_Harris.setArg(1, OFullSize);
        kens->Blackman_Harris.setArg(2, windowSize);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->Blackman_Harris, OFullSize, 64);
    };

    vps.FlatTop = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        kens->FlatTop.setArg(0, *((Buffer*)outReal));
        kens->FlatTop.setArg(1, OFullSize);
        kens->FlatTop.setArg(2, windowSize);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->FlatTop, OFullSize, 64);
    };

    vps.Gaussian = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize, const float sigma){
        kens->Gaussian.setArg(0, *((Buffer*)outReal));
        kens->Gaussian.setArg(1, OFullSize);
        kens->Gaussian.setArg(2, windowSize);
        kens->Gaussian.setArg(3, sigma);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->Gaussian, OFullSize, 64);
    };

    vps.Remove_DC = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        kens->DCRemove.setArg(0, *((Buffer*)outReal));
        kens->DCRemove.setArg(1, OFullSize);
        kens->DCRemove.setArg(2, windowSize);
        clboost::enq_q((*(CommandQueue*)usrPointer), kens->DCRemove, OFullSize, 64);
    };
}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio,
                        const std::string& options)
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
    
    Buffer* Rout = &RealMem;
    Buffer* Iout = &ImagMem;

    std::vector<int> error_container;

    error_container.push_back(kens->Overlap.setArg(0, inMem));
    error_container.push_back(kens->Overlap.setArg(1, OFullSize));
    error_container.push_back(kens->Overlap.setArg(2, FullSize));
    error_container.push_back(kens->Overlap.setArg(3, windowRadix));
    error_container.push_back(kens->Overlap.setArg(4, OMove));
    error_container.push_back(kens->Overlap.setArg(5, RealMem));
    error_container.push_back(clboost::enq_q(CQ, kens->Overlap, OHalfSize, 64));

    vps.UseOption(options, &CQ, &RealMem, OFullSize, windowSize);

    switch (windowRadix)
    {
    case 6:
        error_container.push_back(kens->R6STFT.setArg(0, RealMem));
        error_container.push_back(kens->R6STFT.setArg(1, ImagMem));
        error_container.push_back(kens->R6STFT.setArg(2, OHalfSize));
        error_container.push_back(clboost::enq_q(CQ, kens->R6STFT, OHalfSize, 32));
        break;
    case 7:
        error_container.push_back(kens->R7STFT.setArg(0, RealMem));
        error_container.push_back(kens->R7STFT.setArg(1, ImagMem));
        error_container.push_back(kens->R7STFT.setArg(2, OHalfSize));
        error_container.push_back(clboost::enq_q(CQ, kens->R7STFT, OHalfSize, 64));
        break;
    case 8:
        error_container.push_back(kens->R8STFT.setArg(0, RealMem));
        error_container.push_back(kens->R8STFT.setArg(1, ImagMem));
        error_container.push_back(kens->R8STFT.setArg(2, OHalfSize));
        error_container.push_back(clboost::enq_q(CQ, kens->R8STFT, OHalfSize, 128));
        break;
    case 9:
        error_container.push_back(kens->R9STFT.setArg(0, RealMem));
        error_container.push_back(kens->R9STFT.setArg(1, ImagMem));
        error_container.push_back(kens->R9STFT.setArg(2, OHalfSize));
        error_container.push_back(clboost::enq_q(CQ, kens->R9STFT, OHalfSize, 256));
        break;
    case 10:
        error_container.push_back(kens->R10STFT.setArg(0, RealMem));
        error_container.push_back(kens->R10STFT.setArg(1, ImagMem));
        error_container.push_back(kens->R10STFT.setArg(2, OHalfSize));
        error_container.push_back(clboost::enq_q(CQ, kens->R10STFT, OHalfSize, 512));
        break;
    case 11:
        error_container.push_back(kens->R11STFT.setArg(0, RealMem));
        error_container.push_back(kens->R11STFT.setArg(1, ImagMem));
        error_container.push_back(kens->R11STFT.setArg(2, OHalfSize));
        error_container.push_back(clboost::enq_q(CQ, kens->R11STFT, OHalfSize, 1024));
        break;
    default:
        
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
        if(windowRadix % 2 != 0)
        {
            Rout = &ORealMem;
            Iout = &OImagMem;
        }
        break;
    }

    if(options.find("--half_complex_return") != std::string::npos)
    {
        error_container.push_back(kens->HalfComplex.setArg(0, outMem));
        error_container.push_back(kens->HalfComplex.setArg(1, *Rout));
        error_container.push_back(kens->HalfComplex.setArg(2, *Iout));
        error_container.push_back(kens->HalfComplex.setArg(3, OHalfSize));
        error_container.push_back(kens->HalfComplex.setArg(4, windowRadix));
        error_container.push_back(clboost::enq_q(CQ, kens->HalfComplex, OHalfSize, 32));
    }
    else
    {
        error_container.push_back(kens->toPower.setArg(0, outMem));
        error_container.push_back(kens->toPower.setArg(1, *Rout));
        error_container.push_back(kens->toPower.setArg(2, *Iout));
        error_container.push_back(kens->toPower.setArg(3, OFullSize));
        error_container.push_back(clboost::enq_q(CQ, kens->toPower, OFullSize, 64));
    }
    
    std::vector<float> outData(OFullSize);
    error_container.push_back(clboost::q_read(CQ, outMem, true, OFullSize, outData.data()));
    
    for(auto Eitr : error_container){
        if(Eitr != CL_SUCCESS){
            return std::nullopt;
        }
    }

    return std::move(outData);
}