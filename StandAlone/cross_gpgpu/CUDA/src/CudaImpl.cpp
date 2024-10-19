
#include "cudaStruct.hpp"
#include <functional>
#include <map>
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

void
FillEmbed(std::map<std::string, std::function<const char*()>>& emb)
{
    emb["12690"] = []() -> const char* {
        okl_embed90_12_6 pcode;
        return pcode.ptx_code;
    };
    emb["12680"] = []() -> const char* {
        okl_embed80_12_6 pcode;
        return pcode.ptx_code;
    };
    emb["12675"] = []() -> const char* {
        okl_embed75_12_6 pcode;
        return pcode.ptx_code;
    };
    emb["12670"] = []() -> const char* {
        okl_embed70_12_6 pcode;
        return pcode.ptx_code;
    };
    emb["12661"] = []() -> const char* {
        okl_embed61_12_6 pcode;
        return pcode.ptx_code;
    };
    emb["12652"] = []() -> const char* {
        okl_embed52_12_6 pcode;
        return pcode.ptx_code;
    };
    
    emb["12590"] = []() -> const char* {
        okl_embed90_12_5 pcode;
        return pcode.ptx_code;
    };
    emb["12580"] = []() -> const char* {
        okl_embed80_12_5 pcode;
        return pcode.ptx_code;
    };
    emb["12575"] = []() -> const char* {
        okl_embed75_12_5 pcode;
        return pcode.ptx_code;
    };
    emb["12570"] = []() -> const char* {
        okl_embed70_12_5 pcode;
        return pcode.ptx_code;
    };
    emb["12561"] = []() -> const char* {
        okl_embed61_12_5 pcode;
        return pcode.ptx_code;
    };
    emb["12552"] = []() -> const char* {
        okl_embed52_12_5 pcode;
        return pcode.ptx_code;
    };
    
    emb["12490"] = []() -> const char* {
        okl_embed90_12_4 pcode;
        return pcode.ptx_code;
    };
    emb["12480"] = []() -> const char* {
        okl_embed80_12_4 pcode;
        return pcode.ptx_code;
    };
    emb["12475"] = []() -> const char* {
        okl_embed75_12_4 pcode;
        return pcode.ptx_code;
    };
    emb["12470"] = []() -> const char* {
        okl_embed70_12_4 pcode;
        return pcode.ptx_code;
    };
    emb["12461"] = []() -> const char* {
        okl_embed61_12_4 pcode;
        return pcode.ptx_code;
    };
    emb["12452"] = []() -> const char* {
        okl_embed52_12_4 pcode;
        return pcode.ptx_code;
    };

    emb["12390"] = []() -> const char* {
        okl_embed90_12_3 pcode;
        return pcode.ptx_code;
    };
    emb["12380"] = []() -> const char* {
        okl_embed80_12_3 pcode;
        return pcode.ptx_code;
    };
    emb["12375"] = []() -> const char* {
        okl_embed75_12_3 pcode;
        return pcode.ptx_code;
    };
    emb["12370"] = []() -> const char* {
        okl_embed70_12_3 pcode;
        return pcode.ptx_code;
    };
    emb["12361"] = []() -> const char* {
        okl_embed61_12_3 pcode;
        return pcode.ptx_code;
    };
    emb["12352"] = []() -> const char* {
        okl_embed52_12_3 pcode;
        return pcode.ptx_code;
    };

    emb["12290"] = []() -> const char* {
        okl_embed90_12_2 pcode;
        return pcode.ptx_code;
    };
    emb["12280"] = []() -> const char* {
        okl_embed80_12_2 pcode;
        return pcode.ptx_code;
    };
    emb["12275"] = []() -> const char* {
        okl_embed75_12_2 pcode;
        return pcode.ptx_code;
    };
    emb["12270"] = []() -> const char* {
        okl_embed70_12_2 pcode;
        return pcode.ptx_code;
    };
    emb["12261"] = []() -> const char* {
        okl_embed61_12_2 pcode;
        return pcode.ptx_code;
    };
    emb["12252"] = []() -> const char* {
        okl_embed52_12_2 pcode;
        return pcode.ptx_code;
    };

    emb["12190"] = []() -> const char* {
        okl_embed90_12_1 pcode;
        return pcode.ptx_code;
    };
    emb["12180"] = []() -> const char* {
        okl_embed80_12_1 pcode;
        return pcode.ptx_code;
    };
    emb["12175"] = []() -> const char* {
        okl_embed75_12_1 pcode;
        return pcode.ptx_code;
    };
    emb["12170"] = []() -> const char* {
        okl_embed70_12_1 pcode;
        return pcode.ptx_code;
    };
    emb["12161"] = []() -> const char* {
        okl_embed61_12_1 pcode;
        return pcode.ptx_code;
    };
    emb["12152"] = []() -> const char* {
        okl_embed52_12_1 pcode;
        return pcode.ptx_code;
    };

    emb["12090"] = []() -> const char* {
        okl_embed90_12_0 pcode;
        return pcode.ptx_code;
    };
    emb["12080"] = []() -> const char* {
        okl_embed80_12_0 pcode;
        return pcode.ptx_code;
    };
    emb["12075"] = []() -> const char* {
        okl_embed75_12_0 pcode;
        return pcode.ptx_code;
    };
    emb["12070"] = []() -> const char* {
        okl_embed70_12_0 pcode;
        return pcode.ptx_code;
    };
    emb["12061"] = []() -> const char* {
        okl_embed61_12_0 pcode;
        return pcode.ptx_code;
    };
    emb["12052"] = []() -> const char* {
        okl_embed52_12_0 pcode;
        return pcode.ptx_code;
    };

    emb["11780"] = []() -> const char* {
        okl_embed80_11_7 pcode;
        return pcode.ptx_code;
    };
    emb["11775"] = []() -> const char* {
        okl_embed75_11_7 pcode;
        return pcode.ptx_code;
    };
    emb["11770"] = []() -> const char* {
        okl_embed70_11_7 pcode;
        return pcode.ptx_code;
    };
    emb["11761"] = []() -> const char* {
        okl_embed61_11_7 pcode;
        return pcode.ptx_code;
    };
    emb["11752"] = []() -> const char* {
        okl_embed52_11_7 pcode;
        return pcode.ptx_code;
    };

    emb["11580"] = []() -> const char* {
        okl_embed80_11_5 pcode;
        return pcode.ptx_code;
    };
    emb["11575"] = []() -> const char* {
        okl_embed75_11_5 pcode;
        return pcode.ptx_code;
    };
    emb["11570"] = []() -> const char* {
        okl_embed70_11_5 pcode;
        return pcode.ptx_code;
    };
    emb["11561"] = []() -> const char* {
        okl_embed61_11_5 pcode;
        return pcode.ptx_code;
    };
    emb["11552"] = []() -> const char* {
        okl_embed52_11_5 pcode;
        return pcode.ptx_code;
    };

    emb["11380"] = []() -> const char* {
        okl_embed80_11_3 pcode;
        return pcode.ptx_code;
    };
    emb["11375"] = []() -> const char* {
        okl_embed75_11_3 pcode;
        return pcode.ptx_code;
    };
    emb["11370"] = []() -> const char* {
        okl_embed70_11_3 pcode;
        return pcode.ptx_code;
    };
    emb["11361"] = []() -> const char* {
        okl_embed61_11_3 pcode;
        return pcode.ptx_code;
    };
    emb["11352"] = []() -> const char* {
        okl_embed52_11_3 pcode;
        return pcode.ptx_code;
    };

    emb["11080"] = []() -> const char* {
        okl_embed80_11_0 pcode;
        return pcode.ptx_code;
    };
    emb["11075"] = []() -> const char* {
        okl_embed75_11_0 pcode;
        return pcode.ptx_code;
    };
    emb["11070"] = []() -> const char* {
        okl_embed70_11_0 pcode;
        return pcode.ptx_code;
    };
    emb["11061"] = []() -> const char* {
        okl_embed61_11_0 pcode;
        return pcode.ptx_code;
    };
    emb["11052"] = []() -> const char* {
        okl_embed52_11_0 pcode;
        return pcode.ptx_code;
    };
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
    int cutemp;
    int CUmajor, CUminor;
    int DEVmajor, DEVminor;
    cuDriverGetVersion(&cutemp);
    CUmajor = (cutemp / 1000);
    CUminor = (cutemp % 1000) / 10;
    cuDeviceGetAttribute(&DEVmajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, env->device);
    cuDeviceGetAttribute(&DEVminor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, env->device);
    if(CUmajor == 12){
        CUminor =
        CUminor >= 6 ?
            6
            :
            CUminor
        ;
    }
    if(CUmajor == 11){
        CUminor =
        CUminor >= 7 ?
            7
            :
            CUminor >= 5 ?
                5
                :
                CUminor >= 3 ?
                    3
                    :
                    0
        ;
    }
    if(DEVmajor == 7)
    {
        DEVminor =
        DEVminor >= 5 ?
            5
            :
            0
        ;
    }
    else{
        DEVminor =
        DEVmajor == 9 ?
            0
            :
            DEVmajor == 8 ?
                0
                :
                DEVmajor == 6 ?
                1
                :
                DEVmajor == 5 ?
                2
                :
                -99
        ;
        if(DEVminor == -99){
            exit(1);
        }
    }
    std::string embed_key;
    embed_key += std::to_string(CUmajor);
    embed_key += std::to_string(CUminor);
    embed_key += std::to_string(DEVmajor);
    embed_key += std::to_string(DEVminor);
    
    std::map<std::string, std::function<const char*()>> embeds;
    FillEmbed(embeds);

    if(cuModuleLoadData(&(env->EXPAll), embeds[embed_key]()) != CUDA_SUCCESS){
        exit(1);
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
struct CustomData{
    CUstream* strm;
    const unsigned int* OFullSize;
    const unsigned int* qtC;
    
};
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
    CheckCudaError(cuModuleGetFunction(&(kens->EXPCommon), env->EXPAll, "_occa_StockHamCommon_0"));

    CheckCudaError(cuModuleGetFunction(&(kens->Overlap), env->EXPAll, "_occa_Overlap_Common_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->DCRemove), env->EXPAll, "_occa_DCRemove_Common_0"));

    CheckCudaError(cuModuleGetFunction(&(kens->Hanning), env->EXPAll, "_occa_Window_Hanning_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Hamming), env->EXPAll, "_occa_Window_Hamming_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Blackman), env->EXPAll, "_occa_Window_Blackman_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Nuttall), env->EXPAll, "_occa_Window_Nuttall_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Blackman_Nuttall), env->EXPAll, "_occa_Window_Blackman_Nuttall_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Blackman_Harris), env->EXPAll, "_occa_Window_Blackman_harris_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->FlatTop), env->EXPAll, "_occa_Window_FlatTop_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->Gaussian), env->EXPAll, "_occa_Window_Gaussian_0"));

    CheckCudaError(cuModuleGetFunction(&(kens->toPower), env->EXPAll, "_occa_toPower_0"));
    CheckCudaError(cuModuleGetFunction(&(kens->HalfComplex), env->EXPAll, "_occa_toHalfComplexFormat_0"));
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
        windowSizeEXP,
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