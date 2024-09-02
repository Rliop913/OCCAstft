#include "RunnerInterface.hpp"
#include "compiled.hpp"

struct Genv{
    //no use
};

struct Gcodes{
    //no use
};

void
Runner::InitEnv()
{
    env = nullptr;
    kens = nullptr;
}

void
Runner::UnInit()
{
    
}

void
Runner::BuildKernel()
{
    //no need
}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio)
{
    //default code blocks
    const unsigned int  FullSize    = inData.size();
    const int           windowSize  = 1 << windowRadix;
    const int           qtConst     = toQuot(FullSize, overlapRatio, windowSize);
    const unsigned int  OFullSize   = qtConst * windowSize;
    const unsigned int  OHalfSize   = OFullSize / 2;
    const unsigned int  OMove       = windowSize * (1.0f - overlapRatio);
    //end default

    auto Real = new float  [OFullSize]();
    auto Imag = new float  [OFullSize]();
    

    std::vector<float> outMem(OFullSize);
    
    switch (windowRadix)
    {
    case 10:
        preprocessed_ODW10_STH_STFT
        (
            inData.data(), 
            qtConst, 
            FullSize, 
            OMove,
            OHalfSize,
            Real,
            Imag
        );
        break;
    case 11:
        preprocessed_ODW11_STH_STFT
        (
            inData.data(), 
            qtConst, 
            FullSize, 
            OMove,
            OHalfSize,
            Real,
            Imag
        );
        break;
    default:
        auto OReal = new float  [OFullSize]();
        auto OImag = new float  [OFullSize]();
        Overlap_Common
        (
            inData.data(),
            OFullSize,
            FullSize,
            windowRadix,
            OMove,
            Real
        );
        DCRemove_Common
        (
            Real,
            OFullSize,
            windowSize
        );
        Window_Common
        (
            Real,
            OFullSize,
            windowRadix
        );
        unsigned int HWindowSize = windowSize >> 1;
        for(unsigned int stage = 0; stage < windowRadix; ++stage)
        {
            if(stage % 2 == 0)
            {
                StockHamDITCommon
                (
                    Real,
                    Imag,
                    OReal,
                    OImag,
                    HWindowSize,
                    stage,
                    OHalfSize,
                    windowRadix
                );
            }
            else
            {
                StockHamDITCommon
                (
                    OReal,
                    OImag,
                    Real,
                    Imag,
                    HWindowSize,
                    stage,
                    OHalfSize,
                    windowRadix
                );
            }
        }
        if(windowRadix % 2 == 0)
        {
            toPower
            (
                outMem.data(),
                Real,
                Imag,
                OFullSize,
                windowRadix
            );
        }
        else
        {
            toPower
            (
                outMem.data(),
                OReal,
                OImag,
                OFullSize,
                windowRadix
            );
        }
        delete[] OReal;
        delete[] OImag;
        break;
    }
    delete[] Real;
    delete[] Imag;

    return std::move(outMem);
}