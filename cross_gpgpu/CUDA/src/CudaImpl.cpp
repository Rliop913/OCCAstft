#include "RunnerInterface.hpp"
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
    CUmodule R10;
    CUmodule R11;
    CUmodule R12;
    CUmodule R13;
    CUmodule R14;
    CUmodule R15;
    
    //
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    CUfunction R10STFT;
    CUfunction R11STFT;
    CUfunction R12STFT;
    CUfunction R13STFT;
    CUfunction R14STFT;
    CUfunction R15STFT;
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

int major, minor;
cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, env->device);
cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, env->device);
    std::cout << "Major: " << major << "Minor: " << minor << std::endl;
    CheckCudaError(cuModuleLoad(&(env->R10), "./radix10.ptx"));
    CheckCudaError(cuModuleLoad(&(env->R11), "./radix11.ptx"));
    CheckCudaError(cuModuleLoad(&(env->R12), "./radix12debug.ptx"));
    CheckCudaError(cuModuleLoad(&(env->R13), "./radixCommon.ptx"));
    
    // CheckCudaError(cuModuleLoadDataEx(&(env->R12), "./radix12.ptx",
    // 2, options, optionVals));
    // printf("JIT ERRORLOG: %s\n", error_log);
    // CheckCudaError(cuModuleLoad(&(env->R13), "./radix13.ptx"));
    // CheckCudaError(cuModuleLoad(&(env->R14), "./radix14.ptx"));
    // CheckCudaError(cuModuleLoad(&(env->R15), "./radix14.ptx"));
    
    std::cout<<"CU:41 end init"<<std::endl;
    kens = new Gcodes;
}

void
Runner::UnInit()
{
    cuCtxSynchronize();
    std::cout << "CU:48 end uninit"<<std::endl;
    
    CheckCudaError(cuModuleUnload(env->R10));
    CheckCudaError(cuModuleUnload(env->R11));
    CheckCudaError(cuModuleUnload(env->R12));
    CheckCudaError(cuModuleUnload(env->R13));
    // CheckCudaError(cuModuleUnload(env->R14));
    // CheckCudaError(cuModuleUnload(env->R15));
    std::cout << "CU:51 end uninit"<<std::endl;
    CheckCudaError(cuCtxDestroy(env->context));
    std::cout << "CU:53 end uninit"<<std::endl;

}

// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    CheckCudaError(cuModuleGetFunction(&(kens->R10STFT), env->R10, "_occa_preprocessed_ODW10_STH_STFT_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R11STFT), env->R11, "_occa_preprocessed_ODW11_STH_STFT_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R12STFT), env->R12, "_occa_preprocessed_ODW12_STH_STFT_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R13STFT), env->R13, "_occa_StockHamDITCommon_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R14STFT), env->R13, "_occa_Overlap_Common_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R15STFT), env->R12, "_occa_preprocessed_ODW12_STH_STFT_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->toPower), env->R10, "_occa_toPower_0"));
    
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
    CUdeviceptr DFR;
    CUdeviceptr DFI;
    CUdeviceptr DSR;
    CUdeviceptr DSI;
    
    CheckCudaError(cuMemAllocAsync(&DInput, sizeof(float) * FullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DoverlapBuffer, sizeof(cplx_t) * OFullSize, stream));
    
    CheckCudaError(cuMemAllocAsync(&DOutput, sizeof(float) * OHalfSize, stream));
    
    CheckCudaError(cuMemAllocAsync(&DFR, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DFI, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DSR, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DSI, sizeof(float) * OFullSize, stream));
    
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
    void *pp[] = 
    {
        &DInput,
        (void*)&qtConst,
        (void*)&FullSize,
        (void*)&OMove,
        &DoverlapBuffer
    };
    void *after[] =
    {
        &DoverlapBuffer,
        (void*)&OHalfSize
    };
        // CheckCudaError(cuLaunchKernel(
        //     kens->R13STFT,
        //     qtConst, 1, 1,
        //     1024, 1, 1,
        //     0,
        //     stream,
        //     pp,
        //     NULL
        // ));
        // CheckCudaError(cuLaunchKernel(
        //     kens->R14STFT,
        //     qtConst, 1, 1,
        //     1024, 1, 1,
        //     0,
        //     stream,
        //     after,
        //     NULL
        // ));
        
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
            qtConst / 2, 1, 1,
            1024, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        break;
    case 12:
    std::cout << "CU:171 hit" << std::endl;
        CheckCudaError(cuLaunchKernel(
            kens->R12STFT,
            10,  1, 1,
            1024, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
    
        break;
    // case 13:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R13STFT,
    //         OHalfSize / 4096, 1, 1,
    //         1024, 1, 1,
    //         0,
    //         stream,
    //         AllInOne,
    //         NULL
    //     ));
    //     break;
    // case 14:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R14STFT,
    //         OHalfSize / 8192, 1, 1,
    //         1024, 1, 1,
    //         0,
    //         stream,
    //         AllInOne,
    //         NULL
    //     ));
    //     break;
    // case 15:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R15STFT,
    //         OHalfSize / 16384, 1, 1,
    //         1024, 1, 1,
    //         0,
    //         stream,
    //         AllInOne,
    //         NULL
    //     ));
    //     break;
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
                kens->R14STFT,
                OFullSize / 1024, 1, 1,
                1024, 1, 1,
                0,
                stream,
                overlapCommon,
                NULL
            ));
        std::cout<<"hit default"<<std::endl;
        for (stage = 0; stage < windowRadix; ++stage)
        {
            if (stage % 2 == 0)
            {
                CheckCudaError(cuLaunchKernel(
                    kens->R13STFT,
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
                    kens->R13STFT,
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