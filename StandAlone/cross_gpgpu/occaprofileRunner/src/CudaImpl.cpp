#include "RunnerInterface.hpp"

#include "okl_embed_52_11_6.hpp"
#include "okl_embed_52_12_1.hpp"
#include "okl_embed_52_12_3.hpp"

#include "okl_embed_61_11_6.hpp"
#include "okl_embed_61_12_1.hpp"
#include "okl_embed_61_12_3.hpp"

#include "okl_embed_70_11_6.hpp"
#include "okl_embed_70_12_1.hpp"
#include "okl_embed_70_12_3.hpp"

#include "okl_embed_75_11_6.hpp"
#include "okl_embed_75_12_1.hpp"
#include "okl_embed_75_12_3.hpp"

#include "okl_embed_80_11_6.hpp"
#include "okl_embed_80_12_1.hpp"
#include "okl_embed_80_12_3.hpp"


#include "okl_embed_90_12_1.hpp"
#include "okl_embed_90_12_3.hpp"

#include <cuda.h>
#include "nlohmann/json.hpp"
#include <fstream>

#define LOAD_PTX(buildName, ValueName, IF_Fail_DO)\
    buildName ValueName;\
    if(cuModuleLoadData(&(env->EXPAll), ValueName.ptx_code) != CUDA_SUCCESS)\
    {\
        IF_Fail_DO;\
    }



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
    CUmodule EXPAll;
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    CUfunction EXP6STFT;
    CUfunction EXP7STFT;
    CUfunction EXP8STFT;
    CUfunction EXP9STFT;
    CUfunction EXP10STFT;
    CUfunction EXP11STFT;
    CUfunction EXPCommonOverlap;
    CUfunction EXPCommonSTFT;
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
    std::cout << "major: " << major << "minor: " << minor << std::endl;
    
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
    CheckCudaError(cuModuleUnload(env->EXPAll));
    CheckCudaError(cuCtxDestroy(env->context));

}

// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    CheckCudaError(cuModuleGetFunction(&(kens->EXP6STFT), env->EXPAll, "_occa_Stockhoptimized6_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->EXP7STFT), env->EXPAll, "_occa_Stockhoptimized7_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->EXP8STFT), env->EXPAll, "_occa_Stockhoptimized8_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->EXP9STFT), env->EXPAll, "_occa_Stockhoptimized9_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->EXP10STFT), env->EXPAll, "_occa_Stockhoptimized10_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->EXP11STFT), env->EXPAll, "_occa_Stockhoptimized11_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->EXPCommonOverlap), env->EXPAll, "_occa_Overlap_Common_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->EXPCommonSTFT), env->EXPAll, "_occa_StockHamCommon_0"));
    
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
                        const int& windowSizeEXP, 
                        const float& overlapRatio,
                        const std::string& options)
{
    //default code blocks
    const unsigned int  FullSize    = inData.size();
    const unsigned int  windowSize  = 1 << windowSizeEXP;
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
            (void*)&windowSizeEXP,
            (void*)&OMove,
            &DFR
        };
    CheckCudaError(cuLaunchKernel(
                kens->EXPCommonOverlap,
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
    std::vector<CUevent> starts(windowSizeEXP);
    std::vector<CUevent> ends(windowSizeEXP);
    
    switch (windowSizeEXP)
    {
    case 6:
        cuEventCreate(&(starts[0]), CU_EVENT_DEFAULT);
        cuEventCreate(&(ends[0]), CU_EVENT_DEFAULT);
        cuEventRecord(starts[0], 0);
        CheckCudaError(cuLaunchKernel(
            kens->EXP6STFT,
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
            kens->EXP7STFT,
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
            kens->EXP8STFT,
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
            kens->EXP9STFT,
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
            kens->EXP10STFT,
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
            kens->EXP11STFT,
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
            (void*)&windowSizeEXP,
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
            (void*)&windowSizeEXP,
        };
        
        
        for (stage = 0; stage < windowSizeEXP; ++stage)
        {
            cuEventCreate(&(starts[stage]), CU_EVENT_DEFAULT);
            cuEventCreate(&(ends[stage]), CU_EVENT_DEFAULT);
            cuEventRecord(starts[stage], 0);
            if (stage % 2 == 0)
            {
                CheckCudaError(cuLaunchKernel(
                    kens->EXPCommonSTFT,
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
                    kens->EXPCommonSTFT,
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
        for(int i=0;i<windowSizeEXP;++i)
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