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
    pps.Hanning         = Window_Hanning;
    pps.Hamming         = Window_Hamming;
    pps.Blackman        = Window_Blackman;
    pps.Blackman_Nuttall= Window_Blackman_Nuttall;
    pps.Blackman_Harris = Window_Blackman_harris;
    pps.Nuttall         = Window_Nuttall;
    pps.FlatTop         = Window_Nuttall;
    pps.Gaussian        = Window_Gaussian;
    pps.Remove_DC       = DCRemove_Common;
}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio,
                        const std::string& options)
{
    //default code blocks
    const unsigned int  FullSize    = inData.size();
    const int           windowSize  = 1 << windowRadix;
    const int           qtConst     = toQuot(FullSize, overlapRatio, windowSize);
    const unsigned int  OFullSize   = qtConst * windowSize;
    const unsigned int  OHalfSize   = OFullSize / 2;
    const unsigned int  OMove       = windowSize * (1.0f - overlapRatio);
    //end default



    std::vector<float> Real(OFullSize); 
    std::vector<float> Imag(OFullSize); 
    std::vector<float> outMem(OFullSize);
    
    auto PReal = &Real;
    auto PImag = &Imag;
    
    Overlap_Common
    (
        inData.data(),
        OFullSize,
        FullSize,
        windowRadix,
        OMove,
        Real.data()
    );

    pps.UseOption(options, Real.data(), OFullSize, windowSize);
    
    switch (windowRadix)
    {
    case 6:
        Stockhpotimized6
        (
            Real.data(),
            Imag.data(),
            OHalfSize
        );
        break;
    case 7:
        Stockhpotimized7
        (
            Real.data(),
            Imag.data(),
            OHalfSize
        );
        break;
    case 8:
        Stockhpotimized8
        (
            Real.data(),
            Imag.data(),
            OHalfSize
        );
        break;
    case 9:
        Stockhpotimized9
        (
            Real.data(),
            Imag.data(),
            OHalfSize
        );
        break;
    case 10:
        Stockhpotimized10
        (
            Real.data(),
            Imag.data(),
            OHalfSize
        );
        break;
    case 11:
        Stockhpotimized11
        (
            Real.data(),
            Imag.data(),
            OHalfSize
        );
        break;
    default:
        std::vector<float> OReal(OFullSize);
        std::vector<float> OImag(OFullSize);
        
        unsigned int HWindowSize = windowSize >> 1;
        for(unsigned int stage = 0; stage < windowRadix; ++stage)
        {
            if(stage % 2 == 0)
            {
                StockHamDITCommon
                (
                    Real.data(),
                    Imag.data(),
                    OReal.data(),
                    OImag.data(),
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
                    OReal.data(),
                    OImag.data(),
                    Real.data(),
                    Imag.data(),
                    HWindowSize,
                    stage,
                    OHalfSize,
                    windowRadix
                );
            }
        }
        if(windowRadix % 2 != 0)
        {
            PReal = &OReal;
            PImag = &OImag;
            
        }
        
        break;
    
    }

    if(options.find("--half_complex_return") != std::string::npos)
    {
        toHalfComplexFormat
        (
            (complex *)(outMem.data()),
            (*PReal).data(),
            (*PImag).data(),
            OHalfSize,
            windowRadix
        );
    }
    else
    {
        toPower
        (
            outMem.data(),
            (*PReal).data(),
            (*PImag).data(),
            OFullSize
        );
    }
    return std::move(outMem);
}