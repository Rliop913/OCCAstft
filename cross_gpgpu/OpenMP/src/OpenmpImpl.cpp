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


    auto tempMem = new complex  [OFullSize]();
    auto qtBuffer= new float    [qtConst]();

    std::vector<float> outMem(OHalfSize);

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
            tempMem
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
            tempMem
        );
        break;
    default:
        break;
    }
    
    toPower(tempMem, outMem.data(), OHalfSize, windowRadix);
    delete[] tempMem;
    delete[] qtBuffer;

    return std::move(outMem);
}