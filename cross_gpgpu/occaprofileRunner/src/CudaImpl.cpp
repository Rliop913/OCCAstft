#include "RunnerInterface.hpp"
#include "okl_embed.hpp"
#include <cuda.h>
#include "nlohmann/json.hpp"
#include <fstream>
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
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    CUfunction R6STFT;
    CUfunction R7STFT;
    CUfunction R8STFT;
    CUfunction R9STFT;
    CUfunction R10STFT;
    CUfunction R11STFT;
    CUfunction RadixCommonOverlap;
    CUfunction RadixCommonSTFT;
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
    CheckCudaError(cuModuleGetFunction(&(kens->R6STFT), env->RadixAll, "_occa_Stockhpotimized6_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R7STFT), env->RadixAll, "_occa_Stockhpotimized7_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R8STFT), env->RadixAll, "_occa_Stockhpotimized8_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R9STFT), env->RadixAll, "_occa_Stockhpotimized9_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R10STFT), env->RadixAll, "_occa_Stockhpotimized10_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->R11STFT), env->RadixAll, "_occa_Stockhpotimized11_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->RadixCommonOverlap), env->RadixAll, "_occa_Overlap_Common_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->RadixCommonSTFT), env->RadixAll, "_occa_StockHamDITCommon_0"));
    
}
void
JsonStore
(
    unsigned int windowSize,
    unsigned int BatchSize,
    const std::string& vendor,
    unsigned int NanoSecond
)
{
    using json = nlohmann::json;
    std::ifstream dataFile("./occaResult.json");
    json data = json::parse(dataFile);
    std::string WS = std::to_string(windowSize);
    std::string DS = std::to_string(BatchSize);
    
    std::string vend = WS + vendor + DS;
    
    data[vend] = NanoSecond;
    std::ofstream storeFile("./occaResult.json");
    storeFile << std::setw(4) << data <<std::endl;

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
    
    void *overlapCommon[] =
        {
            &DInput,
            (void*)&OFullSize,
            (void*)&FullSize,
            (void*)&windowRadix,
            (void*)&OMove,
            &DFR
        };
    CheckCudaError(cuLaunchKernel(
                kens->RadixCommonOverlap,
                OFullSize / 64, 1, 1,
                64, 1, 1,
                0,
                stream,
                overlapCommon,
                NULL
            ));
    void *AllInOne[] =
    {
        &DFR,
        &DFI,
        (void*)&OHalfSize
    };
    std::vector<CUevent> starts(windowRadix);
    std::vector<CUevent> ends(windowRadix);
    
    switch (windowRadix)
    {
    case 6:
        cuEventCreate(&(starts[0]), CU_EVENT_DEFAULT);
        cuEventCreate(&(ends[0]), CU_EVENT_DEFAULT);
        cuEventRecord(starts[0], 0);
        CheckCudaError(cuLaunchKernel(
            kens->R6STFT,
            qtConst, 1, 1,
            32, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        cuEventRecord(ends[0], 0);
        
        cuEventSynchronize(ends[0]);
        {
        float milli;
        cuEventElapsedTime(&milli, starts[0], ends[0]);
        unsigned long long nano = milli * 1000000;
        JsonStore(windowSize, qtConst, "occa", nano);
        }
        break;
    case 7:
        cuEventCreate(&(starts[0]), CU_EVENT_DEFAULT);
        cuEventCreate(&(ends[0]), CU_EVENT_DEFAULT);
        cuEventRecord(starts[0], 0);
        CheckCudaError(cuLaunchKernel(
            kens->R7STFT,
            qtConst, 1, 1,
            64, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        cuEventRecord(ends[0], 0);
        
        cuEventSynchronize(ends[0]);
        {
        float milli;
        cuEventElapsedTime(&milli, starts[0], ends[0]);
        unsigned long long nano = milli * 1000000;
        JsonStore(windowSize, qtConst, "occa", nano);
        }
        break;
    case 8:
        cuEventCreate(&(starts[0]), CU_EVENT_DEFAULT);
        cuEventCreate(&(ends[0]), CU_EVENT_DEFAULT);
        cuEventRecord(starts[0], 0);
        CheckCudaError(cuLaunchKernel(
            kens->R8STFT,
            qtConst, 1, 1,
            128, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        cuEventRecord(ends[0], 0);
        
        cuEventSynchronize(ends[0]);
        {
        float milli;
        cuEventElapsedTime(&milli, starts[0], ends[0]);
        unsigned long long nano = milli * 1000000;
        JsonStore(windowSize, qtConst, "occa", nano);
        }
        break;
    case 9:
        cuEventCreate(&(starts[0]), CU_EVENT_DEFAULT);
        cuEventCreate(&(ends[0]), CU_EVENT_DEFAULT);
        cuEventRecord(starts[0], 0);
        CheckCudaError(cuLaunchKernel(
            kens->R9STFT,
            qtConst, 1, 1,
            256, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        cuEventRecord(ends[0], 0);
        
        cuEventSynchronize(ends[0]);
        {
        float milli;
        cuEventElapsedTime(&milli, starts[0], ends[0]);
        unsigned long long nano = milli * 1000000;
        JsonStore(windowSize, qtConst, "occa", nano);
        }
        break;
    case 10:
        cuEventCreate(&(starts[0]), CU_EVENT_DEFAULT);
        cuEventCreate(&(ends[0]), CU_EVENT_DEFAULT);
        cuEventRecord(starts[0], 0);
        CheckCudaError(cuLaunchKernel(
            kens->R10STFT,
            qtConst, 1, 1,
            512, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        cuEventRecord(ends[0], 0);
        
        cuEventSynchronize(ends[0]);
        {
        float milli;
        cuEventElapsedTime(&milli, starts[0], ends[0]);
        unsigned long long nano = milli * 1000000;
        JsonStore(windowSize, qtConst, "occa", nano);
        }
        break;
    case 11:
        cuEventCreate(&(starts[0]), CU_EVENT_DEFAULT);
        cuEventCreate(&(ends[0]), CU_EVENT_DEFAULT);
        cuEventRecord(starts[0], 0);
        CheckCudaError(cuLaunchKernel(
            kens->R11STFT,
            qtConst, 1, 1,
            1024, 1, 1,
            0,
            stream,
            AllInOne,
            NULL
        ));
        cuEventRecord(ends[0], 0);

        cuEventSynchronize(ends[0]);
        {

        float milli;
        cuEventElapsedTime(&milli, starts[0], ends[0]);
        unsigned long long nano = milli * 1000000;
        JsonStore(windowSize, qtConst, "occa", nano);
        }
        break;
    default:
        
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
        
        
        for (stage = 0; stage < windowRadix; ++stage)
        {
            cuEventCreate(&(starts[stage]), CU_EVENT_DEFAULT);
            cuEventCreate(&(ends[stage]), CU_EVENT_DEFAULT);
            cuEventRecord(starts[stage], 0);
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
            cuEventRecord(ends[stage], 0);
        }
        cuEventSynchronize(ends.back());
        {

        float milli = 0;
        for(int i=0;i<windowRadix;++i)
        {
            float local_milli = 0;
            cuEventElapsedTime(&local_milli, starts[i], ends[i]);
            milli += local_milli;
        }

        unsigned long long nano = milli * 1000000;
        JsonStore(windowSize, qtConst, "occa", nano);
        }
        break;
    }
    std::vector<float> outMem(OFullSize);
    
    
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