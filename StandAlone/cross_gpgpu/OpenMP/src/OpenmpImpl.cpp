#include "RunnerInterface.hpp"
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
    
}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowSizeEXP, 
                        const float& overlapRatio,
                        const std::string& options)
{
    //default code blocks
    const unsigned int  FullSize    = inData.size();
    const int           windowSize  = 1 << windowSizeEXP;
    const int           qtConst     = toQuot(FullSize, overlapRatio, windowSize);
    const unsigned int  OFullSize   = qtConst * windowSize;
    const unsigned int  OHalfSize   = OFullSize / 2;
    const unsigned int  OMove       = windowSize * (1.0f - overlapRatio);
    //end default

    std::vector<float> Real(OFullSize); 
    std::vector<float> subReal; 
    std::vector<float> Imag(OFullSize); 
    std::vector<float> subImag;
    std::vector<float> outMem(OFullSize);
    
    runnerFunction::Default_Pipeline
    (
        nullptr,
        &inData,
        &Real,
        &Imag,
        &subReal,
        &subImag,
        &outMem,
        std::move(FullSize),
        std::move(windowSize),
        std::move(qtConst),
        std::move(OFullSize),
        std::move(OHalfSize),
        std::move(OMove),
        options,
        windowSizeEXP,
        overlapRatio
    );


    return std::move(outMem);
}