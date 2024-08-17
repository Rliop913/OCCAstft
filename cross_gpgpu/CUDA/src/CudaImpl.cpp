#include "RunnerInterface.hpp"
#include <cuda.h>
int counter = 0;
void CheckCudaError(CUresult err) {
    ++counter;
    if (err != CUDA_SUCCESS) {
        const char* errorStr = nullptr;
        cuGetErrorString(err, &errorStr);
        std::cerr << counter <<"CUDA Error: " << errorStr << std::endl;
        
    }
}
// Genv: Structure to hold the GPGPU environment settings and resources.
struct Genv{
    CUdevice device;
    CUcontext context;
    CUmodule module;
    //
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    CUfunction overlapNWindow;
    CUfunction rmDC;
    CUfunction bitReverse;
    CUfunction endPreProcess;
    CUfunction butterfly;
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
    CheckCudaError(cuModuleLoad(&(env->module), "./compiled_cuda.ptx"));
    std::cout<<"CU:41 end init"<<std::endl;
    kens = new Gcodes;
}

void
Runner::UnInit()
{
    cuCtxDestroy(env->context);
    cuModuleUnload(env->module);
}

// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    CheckCudaError(cuModuleGetFunction(&(kens->overlapNWindow), env->module, "_occa_overlap_N_window_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->rmDC), env->module, "_occa_removeDC_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->bitReverse), env->module, "_occa_bitReverse_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->endPreProcess), env->module, "_occa_endPreProcess_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->butterfly), env->module, "_occa_Butterfly_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->toPower), env->module, "_occa_toPower_0"));
    std::cout<<"CU:64 end build"<<std::endl;
}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio)
{
    //default code blocks
    const unsigned int  FullSize    = inData.size();
    const int           windowSize  = 1 << windowRadix;
    const int           qtConst     = toQuot(FullSize, overlapRatio, windowSize);//number of windows
    const unsigned int  OFullSize   = qtConst * windowSize; // overlaped fullsize
    const unsigned int  OHalfSize   = OFullSize / 2;
    const unsigned int  OMove       = windowSize * (1.0f - overlapRatio);// window move distance
    //end default
    cuCtxSetCurrent(env->context);
    CUstream stream;
    CheckCudaError(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    std::cout<<"CU:81 end stream init"<<std::endl;
    CUdeviceptr DInput;
    CUdeviceptr DoverlapBuffer;
    CUdeviceptr DqtBuffer;
    CUdeviceptr DOutput;
    CheckCudaError(cuMemAllocAsync(&DInput, sizeof(float) * FullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DoverlapBuffer, sizeof(cplx_t) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DqtBuffer, sizeof(float) * qtConst, stream));
    CheckCudaError(cuMemAllocAsync(&DOutput, sizeof(float) * OHalfSize, stream));
    CheckCudaError(cuMemsetD32Async(DqtBuffer, 0, qtConst, stream));
    CheckCudaError(cuMemcpyHtoDAsync(DInput, inData.data(), sizeof(float) * FullSize, stream));
    ULL FullGridSize = OFullSize / LOCAL_SIZE;
    ULL HalfGridSize = OHalfSize / LOCAL_SIZE;
    void *overlapArgs[] = 
    {
        &DInput, 
        &DoverlapBuffer,
        (void*)&FullSize, 
        (void*)&OFullSize,
        (void*)&windowSize, 
        (void*)&OMove
    };
    CheckCudaError(cuLaunchKernel(
        kens->overlapNWindow,
        FullGridSize, 1, 1,
        LOCAL_SIZE, 1, 1,
        0,
        stream,
        overlapArgs,
        NULL
    ));

    // void *rmdc[] =
    // {
    //     &DoverlapBuffer,
    //     (void*)&OFullSize,
    //     &DqtBuffer,
    //     (void*)&windowSize
    // };
    // CheckCudaError(cuLaunchKernel(
    //     kens->rmDC,
    //     FullGridSize, 1, 1,
    //     LOCAL_SIZE, 1, 1,
    //     0,
    //     stream,
    //     rmdc,
    //     NULL
    // ));

    void *bitRev[] =
    {
        &DoverlapBuffer,
        (void*)&OFullSize,
        (void*)&windowSize,
        (void*)&windowRadix
    };
    CheckCudaError(cuLaunchKernel(
        kens->bitReverse,
        FullGridSize, 1, 1,
        LOCAL_SIZE, 1, 1,
        0,
        stream,
        bitRev,
        NULL
    ));

    void *endPre[] =
    {
        &DoverlapBuffer,
        (void*)&OFullSize
    };
    CheckCudaError(cuLaunchKernel(
        kens->endPreProcess,
        FullGridSize, 1, 1,
        LOCAL_SIZE, 1, 1,
        0,
        stream,
        endPre,
        NULL
    ));

    int powedStage = 0;
    void *butterfly[] =
    {
        &DoverlapBuffer,
        (void*)&windowSize,
        (void*)&powedStage,
        (void*)&OHalfSize,
        (void*)&windowRadix
    };
    for(int iStage=0; iStage < windowRadix; ++iStage)
    {
        powedStage = 1 << iStage;
        CheckCudaError(cuLaunchKernel(
            kens->butterfly,
            HalfGridSize, 1, 1,
            LOCAL_SIZE, 1, 1,
            0,
            stream,
            butterfly,
            NULL
        ));
    }

    void *toPow[] =
    {
        &DoverlapBuffer,
        &DOutput,
        (void*)&OHalfSize,
        (void*)&windowRadix
    };
    CheckCudaError(cuLaunchKernel(
        kens->toPower,
        HalfGridSize, 1, 1,
        LOCAL_SIZE, 1, 1,
        0,
        stream,
        toPow,
        NULL
    ));

    std::vector<float> outMem(OHalfSize);
    cuMemcpyDtoHAsync(outMem.data(), DOutput, OHalfSize * sizeof(float), stream);
    cuStreamSynchronize(stream);
    cuMemFreeAsync(DInput, stream);
    cuMemFreeAsync(DoverlapBuffer, stream);
    cuMemFreeAsync(DqtBuffer, stream);
    cuMemFreeAsync(DOutput, stream);
    cuStreamDestroy(stream);
    
    return std::move(outMem); // If any error occurs during STFT execution, the function returns std::nullopt.
}