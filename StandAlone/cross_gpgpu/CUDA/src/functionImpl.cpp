#include "cudaStruct.hpp"


#define EC_CHECK(N)\
for(int i=0; i < N; ++i)\
{\
    if(EC[i] != CUDA_SUCCESS)\
    {\
        return std::move(false);\
    }\
}\
return std::move(true);


bool 
runnerFunction::Overlap(
    void* userStruct, 
    void* origin, 
    CUI OFullSize, 
    CUI FullSize, 
    CUI windowRadix, 
    CUI OMove, 
    void* Realout
    )
{
    cudaData* Dp = (cudaData*)userStruct;

    void * args[] =
    {
        origin,
        (void*)&OFullSize,
        (void*)&FullSize,
        (void*)&windowRadix,
        (void*)&OMove,
        Realout
    };
    if(
        cuLaunchKernel(
            Dp->kens->Overlap,
            Dp->qtConst, 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}



bool 
runnerFunction::Hanning(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize        
    };
    if(
        cuLaunchKernel(
            Dp->kens->Hanning,
            (OFullSize / 64), 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Hamming(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    cudaData* Dp = (cudaData*)userStruct;

    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize        
    };
    if(
        cuLaunchKernel(
            Dp->kens->Hamming,
            (OFullSize / 64), 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Blackman(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize        
    };
    if(
        cuLaunchKernel(
            Dp->kens->Blackman,
            (OFullSize / 64), 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Nuttall(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize        
    };
    if(
        cuLaunchKernel(
            Dp->kens->Nuttall,
            (OFullSize / 64), 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Blackman_Nuttall(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize        
    };
    if(
        cuLaunchKernel(
            Dp->kens->Blackman_Nuttall,
            (OFullSize / 64), 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Blackman_Harris(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize        
    };
    if(
        cuLaunchKernel(
            Dp->kens->Blackman_Harris,
            (OFullSize / 64), 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::FlatTop(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize        
    };
    if(
        cuLaunchKernel(
            Dp->kens->FlatTop,
            (OFullSize / 64), 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::RemoveDC(void* userStruct, void* data, CUI qtConst, CUI OFullSize, CUI windowSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize        
    };
    if(
        cuLaunchKernel(
            Dp->kens->DCRemove,
            Dp->qtConst, 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Gaussian(
    void* userStruct, 
    void* data, 
    CUI OFullSize, 
    CUI windowSize, 
    const float sigma
    )
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        data,
        (void*)&OFullSize,
        (void*)&windowSize,
        (void*)&sigma
    };
    if(
        cuLaunchKernel(
            Dp->kens->Gaussian,
            (OFullSize / 64), 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}


bool 
runnerFunction::Radix6(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        Real,
        Imag,
        (void*)&OHalfSize
    };
    if(
        cuLaunchKernel(
            Dp->kens->R6STFT,
            Dp->qtConst, 1, 1,
            32, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Radix7(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        Real,
        Imag,
        (void*)&OHalfSize
    };
    if(
        cuLaunchKernel(
            Dp->kens->R7STFT,
            Dp->qtConst, 1, 1,
            64, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Radix8(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        Real,
        Imag,
        (void*)&OHalfSize
    };
    if(
        cuLaunchKernel(
            Dp->kens->R8STFT,
            Dp->qtConst, 1, 1,
            128, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Radix9(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        Real,
        Imag,
        (void*)&OHalfSize
    };
    if(
        cuLaunchKernel(
            Dp->kens->R9STFT,
            Dp->qtConst, 1, 1,
            256, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Radix10(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        Real,
        Imag,
        (void*)&OHalfSize
    };
    if(
        cuLaunchKernel(
            Dp->kens->R10STFT,
            Dp->qtConst, 1, 1,
            512, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

bool 
runnerFunction::Radix11(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    cudaData* Dp = (cudaData*)userStruct;
    
    void * args[] =
    {
        Real,
        Imag,
        (void*)&OHalfSize
    };
    if(
        cuLaunchKernel(
            Dp->kens->R11STFT,
            Dp->qtConst, 1, 1,
            1024, 1, 1,
            0,
            *(Dp->strm),
            args,
            NULL
        ) != CUDA_SUCCESS
    )
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}


bool 
runnerFunction::RadixC(
    void*   userStruct,
    void*   real, 
    void*   imag,
    void*   subreal,
    void*   subimag,
    void*   out,
    CUI     HWindowSize,
    CUI     windowRadix,
    CUI     OFullSize,
    void*   realResult,
    void*   imagResult
    )
{
    cudaData* Dp = (cudaData*)userStruct;
    std::vector<int> EC;
    EC.push_back(cuMemAllocAsync((CUdeviceptr*)subreal, sizeof(float) * OFullSize, *(Dp->strm)));
    EC.push_back(cuMemAllocAsync((CUdeviceptr*)subimag, sizeof(float) * OFullSize, *(Dp->strm)));
    CUI OHalfSize = OFullSize >> 1;
    unsigned int stage =0;
    void *FTSstockham[] =
    {
        real,
        imag,
        subreal,
        subimag,
        (void*)&HWindowSize,
        (void*)&stage,
        (void*)&OHalfSize,
        (void*)&windowRadix,
    };
    void *STFstockham[] =
    {
        subreal,
        subimag,
        real,
        imag,
        (void*)&HWindowSize,
        (void*)&stage,
        (void*)&OHalfSize,
        (void*)&windowRadix,
    };
    for(stage = 0; stage < windowRadix; ++stage)
    {
        if (stage % 2 == 0)
        {
            EC.push_back(cuLaunchKernel(
                Dp->kens->RadixCommon,
                OHalfSize / 256, 1, 1,
                256, 1, 1,
                0,
                *(Dp->strm),
                FTSstockham,
                NULL
            ));
        }
        else
        {
            EC.push_back(cuLaunchKernel(
                Dp->kens->RadixCommon,
                OHalfSize / 256, 1, 1,
                256, 1, 1,
                0,
                *(Dp->strm),
                STFstockham,
                NULL
            ));
        }
    }
    if(windowRadix % 2 != 0)
    {
        realResult = subreal;
        imagResult = subimag;
    }
    EC_CHECK(EC.size());
}


bool 
runnerFunction::HalfComplex(   
    void*   userStruct, 
    void*   out, 
    void*   realResult, 
    void*   imagResult, 
    CUI     OHalfSize, 
    CUI     windowRadix
    )
{
    cudaData* Dp = (cudaData*)userStruct;
    void *halfComplexArgs[] =
    {
        out,
        realResult,
        imagResult,
        (void*)&OHalfSize,
        (void*)&windowRadix
    };
    if(cuLaunchKernel(
        Dp->kens->HalfComplex,
        OHalfSize / 32, 1, 1,
        32, 1, 1,
        0,
        *(Dp->strm),
        halfComplexArgs,
        NULL
    ) != CUDA_SUCCESS)
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}


bool 
runnerFunction::ToPower(   
    void* userStruct, 
    void* out, 
    void* realResult, 
    void* imagResult, 
    CUI OFullSize
    )
{
    cudaData* Dp = (cudaData*)userStruct;
    void *args[] =
    {
        out,
        realResult,
        imagResult,
        (void*)&OFullSize
    };
    if(cuLaunchKernel(
        Dp->kens->toPower,
        (OFullSize / 64), 1, 1,
        64, 1, 1,
        0,
        *(Dp->strm),
        args,
        NULL
    ) != CUDA_SUCCESS)
    {
        return std::move(false);
    }
    else
    {
        return std::move(true);
    }
}

