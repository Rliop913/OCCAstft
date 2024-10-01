
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

    int ec[15];
    cuCtxSetCurrent(env->context);
    CUstream stream;
    ec[0] = (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    std::vector<float> outMem(OFullSize);
    CUdeviceptr DInput;
    CUdeviceptr DOutput;

    CUdeviceptr DFR;
    CUdeviceptr DFI;
    CUdeviceptr DSR;
    CUdeviceptr DSI;

    CUdeviceptr* Rout = &DFR;
    CUdeviceptr* Iout = &DFI;
    
    
    ec[1] = (cuMemAllocAsync(&DInput, sizeof(float) * FullSize, stream));
    ec[2] = (cuMemAllocAsync(&DOutput, sizeof(float) * OFullSize, stream));
    
    ec[3] = (cuMemAllocAsync(&DFR, sizeof(float) * OFullSize, stream));
    ec[4] = (cuMemAllocAsync(&DFI, sizeof(float) * OFullSize, stream));
    
    ec[5] = (cuMemsetD32Async(DFI, 0, OFullSize, stream));
    ec[6] = (cuMemcpyHtoDAsync(DInput, inData.data(), sizeof(float) * FullSize, stream));
    
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
        FullSize,
        windowSize,
        qtConst,
        OFullSize,
        OHalfSize,
        OMove,
        options,
        windowRadix,
        overlapRatio
    );

    
    
    
    ec[7] = (cuMemcpyDtoHAsync(outMem.data(), DOutput, OFullSize * sizeof(float), stream));
    ec[8] = (cuStreamSynchronize(stream));

    ec[9] = (cuMemFreeAsync(DInput, stream));
    ec[10] = (cuMemFreeAsync(DFR, stream));
    ec[11] = (cuMemFreeAsync(DFI, stream));
    {
        if(cuPointerGetAttribute(NULL, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, DSR) == CUDA_SUCCESS)
        {
            CheckCudaError(cuMemFreeAsync(DSR, stream));
            CheckCudaError(cuMemFreeAsync(DSI, stream));
        }
    }
    ec[12] = (cuMemFreeAsync(DOutput, stream));
    ec[13] = (cuStreamSynchronize(stream));
    ec[14] = (cuStreamDestroy(stream));

    if(strRes != "OK")
    {
        std::cerr<< "Err on" << strRes<< std::endl;
        return std::nullopt;
    }
    for(int i=0; i<15; ++i)
    {
        if(ec[i] != CUDA_SUCCESS)
        {
            return std::nullopt;
        }
    }
    return std::move(outMem); // If any error occurs during STFT execution, the function returns std::nullopt.
}