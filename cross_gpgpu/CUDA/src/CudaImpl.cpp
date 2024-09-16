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


#define LOAD_PTX(buildName, ValueName, IF_Fail_DO)\
    buildName ValueName;\
    if(cuModuleLoadData(&(env->RadixAll), ValueName.ptx_code) != CUDA_SUCCESS)\
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
    CUmodule RadixAll;
    
    
    //
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    CUfunction R6STFT;
    CUfunction R7STFT;
    CUfunction R8STFT;
    CUfunction R9STFT;
    CUfunction R10STFT;
    CUfunction R11STFT;
    
    CUfunction RadixCommon;
    CUfunction Overlap;
    CUfunction DCRemove;

    CUfunction Hanning;
    CUfunction Hamming;
    CUfunction Blackman;
    CUfunction Nuttall;
    CUfunction Blackman_Nuttall;
    CUfunction Blackman_Harris;
    CUfunction FlatTop;
    CUfunction Gaussian;

    CUfunction HalfComplex;
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
    const unsigned int* qtCp;
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
    

    vps.Hanning = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->Hanning,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };

    vps.Hamming = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->Hamming,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };
    vps.Blackman = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->Blackman,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };

    vps.Nuttall = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->Nuttall,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };

    vps.Blackman_Nuttall = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->Blackman_Nuttall,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };

    vps.Blackman_Harris = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->Blackman_Harris,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };

    vps.FlatTop = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->FlatTop,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };

    vps.Gaussian = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize, const float sigma){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize,
            (void*)&sigma
        };
        CheckCudaError(cuLaunchKernel(
            kens->Gaussian,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };

    vps.Remove_DC = [this](void *usrPointer, void* outReal, const unsigned int OFullSize, const unsigned int windowSize){
        void *args[] =
        {
            outReal,
            (void*)&OFullSize,
            (void*)&windowSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->DCRemove,
            *(((CustomData*)usrPointer)->qtCp), 1, 1,
            64, 1, 1,
            0,
            *(((CustomData*)usrPointer)->strm),
            args,
            NULL
        ));
    };
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
    bool DS_FLAG = false;

    CUdeviceptr* Rout = &DFR;
    CUdeviceptr* Iout = &DFI;
    
    
    CheckCudaError(cuMemAllocAsync(&DInput, sizeof(float) * FullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DOutput, sizeof(float) * OFullSize, stream));
    
    CheckCudaError(cuMemAllocAsync(&DFR, sizeof(float) * OFullSize, stream));
    CheckCudaError(cuMemAllocAsync(&DFI, sizeof(float) * OFullSize, stream));
    
    CheckCudaError(cuMemsetD32Async(DFI, 0, OFullSize, stream));
    CheckCudaError(cuMemcpyHtoDAsync(DInput, inData.data(), sizeof(float) * FullSize, stream));
    
    
    void *overlapArgs[] =
    {
        &DInput,
        (void*)&OFullSize,
        (void*)&FullSize,
        (void*)&windowRadix,
        (void*)&OMove,
        &DFR,
    };
    CheckCudaError(cuLaunchKernel(
        kens->Overlap,
        qtConst, 1, 1,
        64, 1, 1,
        0,
        stream,
        overlapArgs,
        NULL
    ));
    CustomData cd;
    cd.strm = &stream;
    cd.qtCp = &qtConst;
    vps.UseOption(options, &cd, &DFR, OFullSize, windowSize);

    void *optRadixArgs[] =
    {
        &DFR,
        &DFI,
        (void*)&OHalfSize
    };

    switch (windowRadix)
    {
    case 6:
        CheckCudaError(cuLaunchKernel(
            kens->R6STFT,
            qtConst, 1, 1,
            32, 1, 1,
            0,
            stream,
            optRadixArgs,
            NULL
        ));
        break;
    case 7:
        CheckCudaError(cuLaunchKernel(
            kens->R7STFT,
            qtConst, 1, 1,
            64, 1, 1,
            0,
            stream,
            optRadixArgs,
            NULL
        ));
        break;
    case 8:
        CheckCudaError(cuLaunchKernel(
            kens->R8STFT,
            qtConst, 1, 1,
            128, 1, 1,
            0,
            stream,
            optRadixArgs,
            NULL
        ));
        break;
    case 9:
        CheckCudaError(cuLaunchKernel(
            kens->R9STFT,
            qtConst, 1, 1,
            256, 1, 1,
            0,
            stream,
            optRadixArgs,
            NULL
        ));
        break;
    case 10:
        CheckCudaError(cuLaunchKernel(
            kens->R10STFT,
            qtConst, 1, 1,
            512, 1, 1,
            0,
            stream,
            optRadixArgs,
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
            optRadixArgs,
            NULL
        ));
        break;
    default:
        DS_FLAG = true;
        CheckCudaError(cuMemAllocAsync(&DSR, sizeof(float) * OFullSize, stream));
        CheckCudaError(cuMemAllocAsync(&DSI, sizeof(float) * OFullSize, stream));
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
            if (stage % 2 == 0)
            {
                CheckCudaError(cuLaunchKernel(
                    kens->RadixCommon,
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
                    kens->RadixCommon,
                    OHalfSize / 256, 1, 1,
                    256, 1, 1,
                    0,
                    stream,
                    STFstockham,
                    NULL
                ));
            }
            if(windowRadix % 2 != 0)
            {
                Rout = &DSR;
                Iout = &DSI;
            }
        }
        break;
    }
    std::vector<float> outMem(OFullSize);
    if(options.find("--half_complex_return") != std::string::npos)
    {
        void *halfComplexArgs[] =
        {
            &DOutput,
            Rout,
            Iout,
            (void*)&OHalfSize,
            (void*)&windowRadix
        };
        CheckCudaError(cuLaunchKernel(
            kens->HalfComplex,
            OHalfSize / 32, 1, 1,
            32, 1, 1,
            0,
            stream,
            halfComplexArgs,
            NULL
        ));
    }
    else
    {
        void *powerArgs[] =
        {
            &DOutput,
            Rout,
            Iout,
            (void*)&OFullSize
        };
        CheckCudaError(cuLaunchKernel(
            kens->toPower,
            OFullSize / 64, 1, 1,
            64, 1, 1,
            0,
            stream,
            powerArgs,
            NULL
        ));
    }

    
    CheckCudaError(cuMemcpyDtoHAsync(outMem.data(), DOutput, OFullSize * sizeof(float), stream));
    CheckCudaError(cuStreamSynchronize(stream));

    CheckCudaError(cuMemFreeAsync(DInput, stream));
    CheckCudaError(cuMemFreeAsync(DFR, stream));
    CheckCudaError(cuMemFreeAsync(DFI, stream));
    if (DS_FLAG)
    {
        CheckCudaError(cuMemFreeAsync(DSR, stream));
        CheckCudaError(cuMemFreeAsync(DSI, stream));
    }
    CheckCudaError(cuMemFreeAsync(DOutput, stream));
    CheckCudaError(cuStreamSynchronize(stream));
    CheckCudaError(cuStreamDestroy(stream));
    return std::move(outMem); // If any error occurs during STFT execution, the function returns std::nullopt.
}