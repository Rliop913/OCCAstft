#include "RunnerInterface.hpp"
#include "compiled_serial.hpp"

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

    overlap_N_window(inData.data(), tempMem, FullSize, OFullSize, windowSize, OMove);
    removeDC(tempMem, OFullSize, qtBuffer, windowSize);
    bitReverse(tempMem, OFullSize, windowSize, windowRadix);
    endPreProcess(tempMem, OFullSize);

    for(int iStage=0; iStage < windowRadix; ++iStage)
    {
        Butterfly(tempMem, windowSize, 1<<iStage, OHalfSize, windowRadix);
    }
    toPower(tempMem, outMem.data(), OHalfSize, windowRadix);
    delete[] tempMem;
    delete[] qtBuffer;

    return std::move(outMem);
}