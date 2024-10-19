#include "RunnerInterface.hpp"

bool 
runnerFunction::Overlap(
    void* userStruct, 
    void* origin, 
    CUI OFullSize, 
    CUI FullSize, 
    CUI windowSizeEXP, 
    CUI OMove, 
    void* Realout
    )
{
    return false;
    
}



bool 
runnerFunction::Hanning(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    return false;
    
}

bool 
runnerFunction::Hamming(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    return false;
    
}

bool 
runnerFunction::Blackman(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    return false;
    
}

bool 
runnerFunction::Nuttall(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    return false;
    
}

bool 
runnerFunction::Blackman_Nuttall(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    return false;
    
}

bool 
runnerFunction::Blackman_Harris(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    return false;
    
}

bool 
runnerFunction::FlatTop(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    return false;
    
}

bool 
runnerFunction::RemoveDC(void* userStruct, void* data, CUI qtConst, CUI OFullSize, CUI windowSize)
{
    return false;
    
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
    return false;
    
}


bool 
runnerFunction::EXP6(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    return false;
    
}

bool 
runnerFunction::EXP7(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    return false;
    
}

bool 
runnerFunction::EXP8(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    return false;

}

bool 
runnerFunction::EXP9(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    return false;

}

bool 
runnerFunction::EXP10(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    return false;

}

bool 
runnerFunction::EXP11(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    return false;

}


bool 
runnerFunction::EXPC(
    void*   userStruct,
    void*   real, 
    void*   imag,
    void*   subreal,
    void*   subimag,
    void*   out,
    CUI     HWindowSize,
    CUI     windowSizeEXP,
    CUI     OFullSize,
    void*   realResult,
    void*   imagResult
    )
{
    return false;
    
}


bool 
runnerFunction::HalfComplex(   
    void*   userStruct, 
    void*   out, 
    void*   realResult, 
    void*   imagResult, 
    CUI     OHalfSize, 
    CUI     windowSizeEXP
    )
{
    return false;
    
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
    return false;
    
}

