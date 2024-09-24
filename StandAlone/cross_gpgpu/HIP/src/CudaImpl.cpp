#include "RunnerInterface.hpp"
#include "okl_embed.hpp"
#include <cuda.h>


int counter = 0;
void CheckCudaError(CUresult err) {
    ++counter;
    if (err != CUDA_SUCCESS) {
        const char* errorStr = nullptr;
        const char* errorname = nullptr;
        cuGetErrorString(err, &errorStr);
        cuGetErrorName(err, &errorname);
        
        std::cerr << counter <<"CUDA Error: " << errorStr <<"-"<<errorname << std::endl;
        
        
    }
}
// Genv: Structure to hold the GPGPU environment settings and resources.
struct Genv{
    CUdevice device;
    CUcontext context;
    CUmodule RadixAll;
    
    
    //
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    CUfunction R10STFT;
    CUfunction R11STFT;
    CUfunction RadixCommonOverlap;
    CUfunction RadixCommonRemoveDC;
    CUfunction RadixCommonWindowing;
    CUfunction RadixCommonSTFT;
    CUfunction toPower;
};

// InitEnv: Initializes the GPGPU environment and kernel code structures.
// Allocates memory for 'env' (Genv) and 'kens' (Gcodes).
void
Runner::InitEnv()
{
    env = new Genv;
    CheckCudaError(cuInit(0));
    CheckCudaError(cuDeviceGet(&(env->device), 0));
    CheckCudaError(cuCtxCreate(&(env->context), 0, env->device));
    okl_embed okl;
    CheckCudaError(cuModuleLoadData(&(env->RadixAll), okl.ptx_code));
    kens = new Gcodes;
}

void
Runner::UnInit()
{
    cuCtxSynchronize();
    CheckCudaError(cuModuleUnload(env->RadixAll));
    CheckCudaError(cuCtxDestroy(env->context));

}

// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    CheckCudaError(cuModuleGetFunction(&(kens->R10STFT), env->RadixAll, "_occa_preprocessed_ODW10_STH_STFT_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R11STFT), env->RadixAll, "_occa_preprocessed_ODW11_STH_STFT_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->RadixCommonOverlap), env->RadixAll, "_occa_Overlap_Common_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->RadixCommonRemoveDC), env->RadixAll, "_occa_DCRemove_Common_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->RadixCommonWindowing), env->RadixAll, "_occa_Window_Common_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->RadixCommonSTFT), env->RadixAll, "_occa_StockHamDITCommon_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->toPower), env->RadixAll, "_occa_toPower_0"));
    
}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio)
{
    //default code blocks
    const unsigned int  FullSize    = inData.size();
    const unsigned int  windowSize  = 1 << windowRadix;
    const unsigned int  qtConst     = toQuot(FullSize, overlapRatio, windowSize);//number of windows
    const unsigned int  OFullSize   = qtConst * windowSize; // overlaped fullsize
    const unsigned int  OHalfSize   = OFullSize / 2;
    const unsigned int  OMove       = windowSize * (1.0f - overlapRatio);// window move distance
    //end default
    cuCtxSetCurrent(env->context);
    CUstream stream;
    CheckCudaError(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CUdeviceptr DInput;
    CUdeviceptr DOutput;
    CUdeviceptr DFR;
    CUdeviceptr DFI;
    CUdeviceptr DSR;
    CUdeviceptr DSI;
    
    CheckCudaError(cuMemAllocAsync(&DInput, sizeof(float) * FullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DOutput, sizeof(float) * OFullSize, stream));
    
    CheckCudaError(cuMemAllocAsync(&DFR, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DFI, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DSR, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DSI, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemsetD32Async(DFI, 0, OFullSize, stream));
    CheckCudaError(cuMemcpyHtoDAsync(DInput, inData.data(), sizeof(float) * FullSize, stream));
    
    
    void *AllInOne[] =
    {
        &DInput,
        (void*)&qtConst,
        (void*)&FullSize,
        (void*)&OMove,
        (void*)&OHalfSize,
        &DFR,
        &DFI
    };
        
    switch (windowRadix)
    {
    case 10:
        CheckCudaError(cuLaunchKernel(
            kens->R10STFT,
            qtConst, 1, 1,
            512, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        break;
    case 11:
        CheckCudaError(cuLaunchKernel(
            kens->R11STFT,
            qtConst, 1, 1,
            1024, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        break;
    default:
        void *overlapCommon[] =
        {
            &DInput,
            (void*)&OFullSize,
            (void*)&FullSize,
            (void*)&windowRadix,
            (void*)&OMove,
            &DFR
        };
        void *windowCommon[] =
        {
            &DFR,
            (void*)&OFullSize,
            (void*)&windowRadix
        };
        void *removeDCCommon[] =
        {
            &DFR,
            (void*)&OFullSize,
            (void*)&windowSize,
            &DSI
        };
        auto HwindowSize = windowSize >> 1;
        unsigned int stage =0;
        void *FTSstockham[] =
        {
            &DFR,
            &DFI,
            &DSR,
            &DSI,
            (void*)&HwindowSize,
            (void*)&stage,
            (void*)&OHalfSize,
            (void*)&windowRadix,
        };
        void *STFstockham[] =
        {
            &DSR,
            &DSI,
            &DFR,
            &DFI,
            (void*)&HwindowSize,
            (void*)&stage,
            (void*)&OHalfSize,
            (void*)&windowRadix,
        };
        
        CheckCudaError(cuLaunchKernel(
                kens->RadixCommonOverlap,
                OFullSize / 1024, 1, 1,
                1024, 1, 1,
                0,
                stream,
                overlapCommon,
                NULL
            ));
        CheckCudaError(cuLaunchKernel(
                kens->RadixCommonRemoveDC,
                qtConst, 1, 1,
                64, 1, 1,
                0,
                stream,
                removeDCCommon,
                NULL
            ));
        CheckCudaError(cuLaunchKernel(
                kens->RadixCommonWindowing,
                qtConst, 1, 1,
                1024, 1, 1,
                0,
                stream,
                windowCommon,
                NULL
            ));
        for (stage = 0; stage < windowRadix; ++stage)
        {
            if (stage % 2 == 0)
            {
                CheckCudaError(cuLaunchKernel(
                    kens->RadixCommonSTFT,
                    OHalfSize / 256, 1, 1,
                    256, 1, 1,
                    0,
                    stream,
                    FTSstockham,
                    NULL
                ));
            }
            else
            {
                CheckCudaError(cuLaunchKernel(
                    kens->RadixCommonSTFT,
                    OHalfSize / 256, 1, 1,
                    256, 1, 1,
                    0,
                    stream,
                    STFstockham,
                    NULL
                ));
            }
        }
        break;
    }
    std::vector<float> outMem(OFullSize);
    void *powerArg[5];
    powerArg[0] = &DOutput;
    powerArg[3] = (void*)&OFullSize;
    powerArg[4] = (void*)&windowRadix;
    if (windowRadix % 2 == 0)
    {
        powerArg[1] = &DFR;
        powerArg[2] = &DFI;       
    }
    else
    {
        powerArg[1] = &DSR;
        powerArg[2] = &DSI;
    }
    
    CheckCudaError(cuLaunchKernel(
                    kens->toPower,
                    OFullSize / 256, 1, 1,
                    256, 1, 1,
                    0,
                    stream,
                    powerArg,
                    NULL
                ));

    
    CheckCudaError(cuMemcpyDtoHAsync(outMem.data(), DOutput, OFullSize * sizeof(float), stream));
    CheckCudaError(cuStreamSynchronize(stream));
    CheckCudaError(cuMemFreeAsync(DInput, stream));
    CheckCudaError(cuMemFreeAsync(DFR, stream));
    CheckCudaError(cuMemFreeAsync(DFI, stream));
    CheckCudaError(cuMemFreeAsync(DSR, stream));
    CheckCudaError(cuMemFreeAsync(DSI, stream));
    CheckCudaError(cuMemFreeAsync(DOutput, stream));
    CheckCudaError(cuStreamSynchronize(stream));
    CheckCudaError(cuStreamDestroy(stream));
    return std::move(outMem); // If any error occurs during STFT execution, the function returns std::nullopt.
}