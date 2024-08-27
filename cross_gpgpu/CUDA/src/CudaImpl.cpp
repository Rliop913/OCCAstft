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
    CUfunction R10STFT;
    CUfunction R11STFT;
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
    CheckCudaError(cuModuleLoad(&(env->module), "./compiled.ptx"));
    std::cout<<"CU:41 end init"<<std::endl;
    kens = new Gcodes;
}

void
Runner::UnInit()
{
    cuCtxSynchronize();
    std::cout << "CU:48 end uninit"<<std::endl;
    
    CheckCudaError(cuModuleUnload(env->module));
    std::cout << "CU:51 end uninit"<<std::endl;
    CheckCudaError(cuCtxDestroy(env->context));
    std::cout << "CU:53 end uninit"<<std::endl;

}

// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    CheckCudaError(cuModuleGetFunction(&(kens->R10STFT), env->module, "_occa_preprocessed_ODW10_STH_STFT_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R11STFT), env->module, "_occa_preprocessed_ODW11_STH_STFT_0"));
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
    const unsigned int  windowSize  = 1 << windowRadix;
    const unsigned int  qtConst     = toQuot(FullSize, overlapRatio, windowSize);//number of windows
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
    CUdeviceptr DOutput;
    CheckCudaError(cuMemAllocAsync(&DInput, sizeof(float) * FullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DoverlapBuffer, sizeof(cplx_t) * OFullSize, stream));
    
    CheckCudaError(cuMemAllocAsync(&DOutput, sizeof(float) * OHalfSize, stream));
    
    CheckCudaError(cuMemcpyHtoDAsync(DInput, inData.data(), sizeof(float) * FullSize, stream));
    ULL FullGridSize = OFullSize / LOCAL_SIZE;
    ULL HalfGridSize = OHalfSize / LOCAL_SIZE;
    
    std::cout << "CU:94 end overlap"<<std::endl;
    
    void *AllInOne[] =
    {
        &DInput,
        (void*)&qtConst,
        (void*)&FullSize,
        (void*)&OMove,
        (void*)&OHalfSize,
        &DoverlapBuffer
    };

    switch (windowRadix)
    {
    case 10:
        CheckCudaError(cuLaunchKernel(
            kens->R10STFT,
            OHalfSize / 512, 1, 1,
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
            OHalfSize / 1024, 1, 1,
            1024, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        break;
    default:
        break;
    }


    std::cout << "CU:177 end butterfly"<<std::endl;
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
    std::cout << "CU:194 end pow"<<std::endl;
    std::vector<float> outMem(OHalfSize);
    
    CheckCudaError(cuMemcpyDtoHAsync(outMem.data(), DOutput, OHalfSize * sizeof(float), stream));
    CheckCudaError(cuStreamSynchronize(stream));
    CheckCudaError(cuMemFreeAsync(DInput, stream));
    CheckCudaError(cuMemFreeAsync(DoverlapBuffer, stream));
    CheckCudaError(cuMemFreeAsync(DOutput, stream));
    CheckCudaError(cuStreamSynchronize(stream));
    CheckCudaError(cuStreamDestroy(stream));
    std::cout << "CU:203 end destroy"<<std::endl;
    return std::move(outMem); // If any error occurs during STFT execution, the function returns std::nullopt.
}