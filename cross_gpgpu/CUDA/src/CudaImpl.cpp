
#include "cudaStruct.hpp"

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
    
    switch (major)
    {
    case 5:
        {
            LOAD_PTX(okl_embed52_12_3, k123, LOAD_PTX(okl_embed52_12_1, k121, LOAD_PTX(okl_embed52_11_6, k116, ;)));
        }
        break;
    case 6:
        {
            LOAD_PTX(okl_embed61_12_3, k123, LOAD_PTX(okl_embed61_12_1, k121, LOAD_PTX(okl_embed61_11_6, k116, ;)));
        }
        break;
    case 7:
        if(minor >= 5)
        {
            LOAD_PTX(okl_embed75_12_3, k123, LOAD_PTX(okl_embed75_12_1, k121, LOAD_PTX(okl_embed75_11_6, k116, ;)));
        }
        else
        {
            LOAD_PTX(okl_embed70_12_3, k123, LOAD_PTX(okl_embed70_12_1, k121, LOAD_PTX(okl_embed70_11_6, k116, ;)));
        }
        break;
    case 8:
        {
            LOAD_PTX(okl_embed80_12_3, k123, LOAD_PTX(okl_embed80_12_1, k121, LOAD_PTX(okl_embed80_11_6, k116, ;)));
        }
        break;

    case 9:
        {
            LOAD_PTX(okl_embed90_12_3, k123, LOAD_PTX(okl_embed90_12_1, k121, ;));
        }
        break;
    default:
        break;
    }
    
    kens = new Gcodes;
}

void
Runner::UnInit()
{
    cuCtxSynchronize();
    CheckCudaError(cuModuleUnload(env->RadixAll));
    CheckCudaError(cuCtxDestroy(env->context));

}
struct CustomData{
    CUstream* strm;
    const unsigned int* OFullSize;
    const unsigned int* qtC;
    
};
// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    CheckCudaError(cuModuleGetFunction(&(kens->R6STFT), env->RadixAll, "_occa_Stockhpotimized6_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R7STFT), env->RadixAll, "_occa_Stockhpotimized7_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R8STFT), env->RadixAll, "_occa_Stockhpotimized8_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R9STFT), env->RadixAll, "_occa_Stockhpotimized9_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R10STFT), env->RadixAll, "_occa_Stockhpotimized10_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R11STFT), env->RadixAll, "_occa_Stockhpotimized11_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->RadixCommon), env->RadixAll, "_occa_StockHamDITCommon_0"));

    CheckCudaError(cuModuleGetFunction(&(kens->Overlap), env->RadixAll, "_occa_Overlap_Common_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->DCRemove), env->RadixAll, "_occa_DCRemove_Common_0"));

    CheckCudaError(cuModuleGetFunction(&(kens->Hanning), env->RadixAll, "_occa_Window_Hanning_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Hamming), env->RadixAll, "_occa_Window_Hamming_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Blackman), env->RadixAll, "_occa_Window_Blackman_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Nuttall), env->RadixAll, "_occa_Window_Nuttall_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Blackman_Nuttall), env->RadixAll, "_occa_Window_Blackman_Nuttall_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Blackman_Harris), env->RadixAll, "_occa_Window_Blackman_harris_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->FlatTop), env->RadixAll, "_occa_Window_FlatTop_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Gaussian), env->RadixAll, "_occa_Window_Gaussian_0"));

    CheckCudaError(cuModuleGetFunction(&(kens->toPower), env->RadixAll, "_occa_toPower_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->HalfComplex), env->RadixAll, "_occa_toHalfComplexFormat_0"));
}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio,
                        const std::string& options)
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

    CUdeviceptr* Rout = &DFR;
    CUdeviceptr* Iout = &DFI;
    
    
    CheckCudaError(cuMemAllocAsync(&DInput, sizeof(float) * FullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DOutput, sizeof(float) * OFullSize, stream));
    
    CheckCudaError(cuMemAllocAsync(&DFR, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DFI, sizeof(float) * OFullSize, stream));
    
    CheckCudaError(cuMemsetD32Async(DFI, 0, OFullSize, stream));
    CheckCudaError(cuMemcpyHtoDAsync(DInput, inData.data(), sizeof(float) * FullSize, stream));
    
    cudaData cud;
    cud.env = env;
    cud.kens= kens;
    cud.qtConst = qtConst;
    cud.strm = &stream;
    
    std::string strRes = 
    runnerFunction::Default_Pipeline
    (
        &cud,
        &DInput,
        &DFR,
        &DFI,
        &DSR,
        &DSI,
        &DOutput,
        std::move(FullSize),
        std::move(windowSize),
        std::move(qtConst),
        std::move(OFullSize),
        std::move(OHalfSize),
        std::move(OMove),
        options,
        windowRadix,
        overlapRatio
    );

    
    // void *overlapArgs[] =
    // {
    //     &DInput,
    //     (void*)&OFullSize,
    //     (void*)&FullSize,
    //     (void*)&windowRadix,
    //     (void*)&OMove,
    //     &DFR,
    // };
    // CheckCudaError(cuLaunchKernel(
    //     kens->Overlap,
    //     qtConst, 1, 1,
    //     64, 1, 1,
    //     0,
    //     stream,
    //     overlapArgs,
    //     NULL
    // ));
    // CustomData cd;
    // cd.strm = &stream;
    // cd.OFullSize = &OFullSize;
    // cd.qtC = &qtConst;
    // vps.UseOption(options, &cd, &DFR, OFullSize, windowSize);

    // void *optRadixArgs[] =
    // {
    //     &DFR,
    //     &DFI,
    //     (void*)&OHalfSize
    // };

    // switch (windowRadix)
    // {
    // case 6:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R6STFT,
    //         qtConst, 1, 1,
    //         32, 1, 1,
    //         0,
    //         stream,
    //         optRadixArgs,
    //         NULL
    //     ));
    //     break;
    // case 7:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R7STFT,
    //         qtConst, 1, 1,
    //         64, 1, 1,
    //         0,
    //         stream,
    //         optRadixArgs,
    //         NULL
    //     ));
    //     break;
    // case 8:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R8STFT,
    //         qtConst, 1, 1,
    //         128, 1, 1,
    //         0,
    //         stream,
    //         optRadixArgs,
    //         NULL
    //     ));
    //     break;
    // case 9:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R9STFT,
    //         qtConst, 1, 1,
    //         256, 1, 1,
    //         0,
    //         stream,
    //         optRadixArgs,
    //         NULL
    //     ));
    //     break;
    // case 10:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R10STFT,
    //         qtConst, 1, 1,
    //         512, 1, 1,
    //         0,
    //         stream,
    //         optRadixArgs,
    //         NULL
    //     ));
    //     break;
    // case 11:
    //     CheckCudaError(cuLaunchKernel(
    //         kens->R11STFT,
    //         qtConst, 1, 1,
    //         1024, 1, 1,
    //         0,
    //         stream,
    //         optRadixArgs,
    //         NULL
    //     ));
    //     break;
    // default:
    //     DS_FLAG = true;
    //     CheckCudaError(cuMemAllocAsync(&DSR, sizeof(float) * OFullSize, stream));
    //     CheckCudaError(cuMemAllocAsync(&DSI, sizeof(float) * OFullSize, stream));
    //     auto HwindowSize = windowSize >> 1;
    //     unsigned int stage =0;
    //     void *FTSstockham[] =
    //     {
    //         &DFR,
    //         &DFI,
    //         &DSR,
    //         &DSI,
    //         (void*)&HwindowSize,
    //         (void*)&stage,
    //         (void*)&OHalfSize,
    //         (void*)&windowRadix,
    //     };
    //     void *STFstockham[] =
    //     {
    //         &DSR,
    //         &DSI,
    //         &DFR,
    //         &DFI,
    //         (void*)&HwindowSize,
    //         (void*)&stage,
    //         (void*)&OHalfSize,
    //         (void*)&windowRadix,
    //     };
        
    //     for (stage = 0; stage < windowRadix; ++stage)
    //     {
    //         if (stage % 2 == 0)
    //         {
    //             CheckCudaError(cuLaunchKernel(
    //                 kens->RadixCommon,
    //                 OHalfSize / 256, 1, 1,
    //                 256, 1, 1,
    //                 0,
    //                 stream,
    //                 FTSstockham,
    //                 NULL
    //             ));
    //         }
    //         else
    //         {
    //             CheckCudaError(cuLaunchKernel(
    //                 kens->RadixCommon,
    //                 OHalfSize / 256, 1, 1,
    //                 256, 1, 1,
    //                 0,
    //                 stream,
    //                 STFstockham,
    //                 NULL
    //             ));
    //         }
    //         if(windowRadix % 2 != 0)
    //         {
    //             Rout = &DSR;
    //             Iout = &DSI;
    //         }
    //     }
    //     break;
    // }
    
    // if(options.find("--half_complex_return") != std::string::npos)
    // {
    //     void *halfComplexArgs[] =
    //     {
    //         &DOutput,
    //         Rout,
    //         Iout,
    //         (void*)&OHalfSize,
    //         (void*)&windowRadix
    //     };
    //     CheckCudaError(cuLaunchKernel(
    //         kens->HalfComplex,
    //         OHalfSize / 32, 1, 1,
    //         32, 1, 1,
    //         0,
    //         stream,
    //         halfComplexArgs,
    //         NULL
    //     ));
    // }
    // else
    // {
    //     void *powerArgs[] =
    //     {
    //         &DOutput,
    //         Rout,
    //         Iout,
    //         (void*)&OFullSize
    //     };
    //     CheckCudaError(cuLaunchKernel(
    //         kens->toPower,
    //         OFullSize / 64, 1, 1,
    //         64, 1, 1,
    //         0,
    //         stream,
    //         powerArgs,
    //         NULL
    //     ));
    // }
    std::vector<float> outMem(OFullSize);
    int ec[8];
    ec[0] = (cuMemcpyDtoHAsync(outMem.data(), DOutput, OFullSize * sizeof(float), stream));
    ec[1] = (cuStreamSynchronize(stream));

    ec[2] = (cuMemFreeAsync(DInput, stream));
    ec[3] = (cuMemFreeAsync(DFR, stream));
    ec[4] = (cuMemFreeAsync(DFI, stream));
    {
        if(cuPointerGetAttribute(NULL, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, DSR) == CUDA_SUCCESS)
        {
            CheckCudaError(cuMemFreeAsync(DSR, stream));
            CheckCudaError(cuMemFreeAsync(DSI, stream));
        }
    }
    ec[5] = (cuMemFreeAsync(DOutput, stream));
    ec[6] = (cuStreamSynchronize(stream));
    ec[7] = (cuStreamDestroy(stream));

    if(strRes != "OK")
    {
        return std::nullopt;
    }
    for(int i=0; i<8; ++i)
    {
        if(ec[i] != CUDA_SUCCESS)
        {
            return std::nullopt;
        }
    }
    return std::move(outMem); // If any error occurs during STFT execution, the function returns std::nullopt.
}