#include "clStruct.hpp"


#define EC_CHECK(N)\
for(int i=0; i < N; ++i)\
{\
    if(EC[i] != CL_SUCCESS)\
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
    clData* Dp = (clData*)userStruct;
    int EC[7];
    EC[0] = Dp->kens->Overlap.setArg(0, (*(Buffer*)origin));
    EC[1] = Dp->kens->Overlap.setArg(1, OFullSize);
    EC[2] = Dp->kens->Overlap.setArg(2, FullSize);
    EC[3] = Dp->kens->Overlap.setArg(3, windowRadix);
    EC[4] = Dp->kens->Overlap.setArg(4, OMove);
    EC[5] = Dp->kens->Overlap.setArg(5, (*(Buffer*)Realout));
    EC[6] = clboost::enq_q(*(Dp->cq), Dp->kens->Overlap, OFullSize, 64);

    EC_CHECK(7)
}



bool 
runnerFunction::Hanning(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->Hanning.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->Hanning.setArg(1, OFullSize);
    EC[2] = Dp->kens->Hanning.setArg(2, windowSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->Hanning, OFullSize, 64);

    EC_CHECK(4)
}

bool 
runnerFunction::Hamming(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->Hamming.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->Hamming.setArg(1, OFullSize);
    EC[2] = Dp->kens->Hamming.setArg(2, windowSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->Hamming, OFullSize, 64);

    EC_CHECK(4)
}

bool 
runnerFunction::Blackman(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->Blackman.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->Blackman.setArg(1, OFullSize);
    EC[2] = Dp->kens->Blackman.setArg(2, windowSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->Blackman, OFullSize, 64);

    EC_CHECK(4)
}

bool 
runnerFunction::Nuttall(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->Nuttall.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->Nuttall.setArg(1, OFullSize);
    EC[2] = Dp->kens->Nuttall.setArg(2, windowSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->Nuttall, OFullSize, 64);

    EC_CHECK(4)
}

bool 
runnerFunction::Blackman_Nuttall(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->Blackman_Nuttall.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->Blackman_Nuttall.setArg(1, OFullSize);
    EC[2] = Dp->kens->Blackman_Nuttall.setArg(2, windowSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->Blackman_Nuttall, OFullSize, 64);

    EC_CHECK(4)
}

bool 
runnerFunction::Blackman_Harris(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->Blackman_Harris.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->Blackman_Harris.setArg(1, OFullSize);
    EC[2] = Dp->kens->Blackman_Harris.setArg(2, windowSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->Blackman_Harris, OFullSize, 64);

    EC_CHECK(4)
}

bool 
runnerFunction::FlatTop(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->FlatTop.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->FlatTop.setArg(1, OFullSize);
    EC[2] = Dp->kens->FlatTop.setArg(2, windowSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->FlatTop, OFullSize, 64);

    EC_CHECK(4)
}

bool 
runnerFunction::RemoveDC(void* userStruct, void* data, CUI qtConst, CUI OFullSize, CUI windowSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->DCRemove.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->DCRemove.setArg(1, OFullSize);
    EC[2] = Dp->kens->DCRemove.setArg(2, windowSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->DCRemove, qtConst * 64, 64);

    EC_CHECK(4)
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
    clData* Dp = (clData*)userStruct;
    int EC[5];
    EC[0] = Dp->kens->Gaussian.setArg(0, (*(Buffer*)data));
    EC[1] = Dp->kens->Gaussian.setArg(1, OFullSize);
    EC[2] = Dp->kens->Gaussian.setArg(2, windowSize);
    EC[3] = Dp->kens->Gaussian.setArg(2, sigma);
    EC[4] = clboost::enq_q(*(Dp->cq), Dp->kens->Gaussian, OFullSize, 64);

    EC_CHECK(5)
}


bool 
runnerFunction::Radix6(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->R6STFT.setArg(0, (*(Buffer*)Real));
    EC[1] = Dp->kens->R6STFT.setArg(1, (*(Buffer*)Imag));
    EC[2] = Dp->kens->R6STFT.setArg(2, OHalfSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->R6STFT, OHalfSize, 32);

    EC_CHECK(4)
}

bool 
runnerFunction::Radix7(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->R7STFT.setArg(0, (*(Buffer*)Real));
    EC[1] = Dp->kens->R7STFT.setArg(1, (*(Buffer*)Imag));
    EC[2] = Dp->kens->R7STFT.setArg(2, OHalfSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->R7STFT, OHalfSize, 64);

    EC_CHECK(4)
}

bool 
runnerFunction::Radix8(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->R8STFT.setArg(0, (*(Buffer*)Real));
    EC[1] = Dp->kens->R8STFT.setArg(1, (*(Buffer*)Imag));
    EC[2] = Dp->kens->R8STFT.setArg(2, OHalfSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->R8STFT, OHalfSize, 128);

    EC_CHECK(4)
}

bool 
runnerFunction::Radix9(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->R9STFT.setArg(0, (*(Buffer*)Real));
    EC[1] = Dp->kens->R9STFT.setArg(1, (*(Buffer*)Imag));
    EC[2] = Dp->kens->R9STFT.setArg(2, OHalfSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->R9STFT, OHalfSize, 256);

    EC_CHECK(4)
}

bool 
runnerFunction::Radix10(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->R10STFT.setArg(0, (*(Buffer*)Real));
    EC[1] = Dp->kens->R10STFT.setArg(1, (*(Buffer*)Imag));
    EC[2] = Dp->kens->R10STFT.setArg(2, OHalfSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->R10STFT, OHalfSize, 512);

    EC_CHECK(4)
}

bool 
runnerFunction::Radix11(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    clData* Dp = (clData*)userStruct;
    int EC[4];
    EC[0] = Dp->kens->R11STFT.setArg(0, (*(Buffer*)Real));
    EC[1] = Dp->kens->R11STFT.setArg(1, (*(Buffer*)Imag));
    EC[2] = Dp->kens->R11STFT.setArg(2, OHalfSize);
    EC[3] = clboost::enq_q(*(Dp->cq), Dp->kens->R11STFT, OHalfSize, 1024);

    EC_CHECK(4)
}


bool 
runnerFunction::RadixC(
    void*   userStruct,
    void*   real, 
    void*   imag,
    void*   subreal,
    void*   subimag,
    void*   out,
    CUI&&   HWindowSize,
    CUI     windowRadix,
    CUI     OFullSize,
    void*   realResult,
    void*   imagResult
    )
{
    clData* Dp = (clData*)userStruct;
    std::vector<int> EC;
    *((Buffer*)subreal) = clboost::DMEM<float>(Dp->env->CT, OFullSize);
    *((Buffer*)subimag) = clboost::DMEM<float>(Dp->env->CT, OFullSize);
    CUI OHalfSize = OFullSize >> 1;

    
    for(unsigned int stage = 0; stage < windowRadix; ++stage)
    {
        EC.push_back(Dp->kens->RadixCommon.setArg(5, stage));
        if(stage % 2 == 0)
        {
            EC.push_back(Dp->kens->RadixCommon.setArg(0, (*(Buffer*)real)));
            EC.push_back(Dp->kens->RadixCommon.setArg(1, (*(Buffer*)imag)));
            EC.push_back(Dp->kens->RadixCommon.setArg(2, (*(Buffer*)subreal)));
            EC.push_back(Dp->kens->RadixCommon.setArg(3, (*(Buffer*)subimag)));
        }
        else
        {
            EC.push_back(Dp->kens->RadixCommon.setArg(0, (*(Buffer*)subreal)));
            EC.push_back(Dp->kens->RadixCommon.setArg(1, (*(Buffer*)subimag)));
            EC.push_back(Dp->kens->RadixCommon.setArg(2, (*(Buffer*)real)));
            EC.push_back(Dp->kens->RadixCommon.setArg(3, (*(Buffer*)imag)));
        }
        EC.push_back(clboost::enq_q(*(Dp->cq), Dp->kens->RadixCommon, OHalfSize, 256));
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
    clData* Dp = (clData*)userStruct;
    int EC[6];
    EC[0] = Dp->kens->HalfComplex.setArg(0, (*(Buffer*)out));
    EC[1] = Dp->kens->HalfComplex.setArg(1, (*(Buffer*)realResult));
    EC[2] = Dp->kens->HalfComplex.setArg(2, (*(Buffer*)imagResult));
    EC[3] = Dp->kens->HalfComplex.setArg(3, OHalfSize);
    EC[4] = Dp->kens->HalfComplex.setArg(4, windowRadix);
    EC[5] = clboost::enq_q(*(Dp->cq), Dp->kens->HalfComplex, OHalfSize, 32);

    EC_CHECK(6)
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
    clData* Dp = (clData*)userStruct;
    int EC[5];
    EC[0] = Dp->kens->toPower.setArg(0, (*(Buffer*)out));
    EC[1] = Dp->kens->toPower.setArg(1, (*(Buffer*)realResult));
    EC[2] = Dp->kens->toPower.setArg(2, (*(Buffer*)imagResult));
    EC[3] = Dp->kens->toPower.setArg(3, OFullSize);
    EC[4] = clboost::enq_q(*(Dp->cq), Dp->kens->toPower, OFullSize, 64);

    EC_CHECK(5)
}

