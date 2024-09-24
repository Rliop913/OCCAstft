#include "RunnerInterface.hpp"
#include "compiled.hpp"
extern void Overlap_Common          (float*, CUI&, CUI&, CUI&, CUI&, float*);
extern void Window_Hanning          (float*, CUI&, CUI& );
extern void Window_Hamming          (float*, CUI&, CUI& );
extern void Window_Blackman         (float*, CUI&, CUI& );
extern void Window_Nuttall          (float*, CUI&, CUI& );
extern void Window_Blackman_Hanning (float*, CUI&, CUI& );
extern void Window_Blackman_harris  (float*, CUI&, CUI& );
extern void Window_FlatTop          (float*, CUI&, CUI& );
extern void Window_Gaussian         (float*, CUI&, CUI&, const float& );
extern void Window_Blackman_Nuttall (float*, CUI&, CUI& );
extern void DCRemove_Common         (float*, CUI&, CUI& );

extern void Stockhpotimized6        (float*, float*, CUI&);
extern void Stockhpotimized7        (float*, float*, CUI&);
extern void Stockhpotimized8        (float*, float*, CUI&);
extern void Stockhpotimized9        (float*, float*, CUI&);
extern void Stockhpotimized10       (float*, float*, CUI&);
extern void Stockhpotimized11       (float*, float*, CUI&);
extern void StockHamDITCommon       (float*,
                                     float*,
                                     float*,
                                     float*,
                                     const unsigned int &,
                                     const unsigned int &,
                                     const unsigned int &,
                                     const unsigned int &);

extern void toPower                 (float*,
                                     float*,
                                     float*,
                                     const unsigned int &);

extern void toHalfComplexFormat     (complex*,
                                     float*,
                                     float*,
                                     const unsigned int &,
                                     const int &);

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
    Overlap_Common
    (
        ((std::vector<float>*)origin)->data(),
        OFullSize,
        FullSize,
        windowRadix,
        OMove,
        ((std::vector<float>*)Realout)->data()
    );
    return std::move(true);
}

bool 
runnerFunction::Hanning(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    Window_Hanning(((std::vector<float>*)data)->data(), OFullSize, windowSize);
    return std::move(true);
}

bool 
runnerFunction::Hamming(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    Window_Hamming(((std::vector<float>*)data)->data(), OFullSize, windowSize);
    return std::move(true);
}

bool 
runnerFunction::Blackman(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    Window_Blackman(((std::vector<float>*)data)->data(), OFullSize, windowSize);
    return std::move(true);
}

bool 
runnerFunction::Nuttall(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    Window_Nuttall(((std::vector<float>*)data)->data(), OFullSize, windowSize);
    return std::move(true);
}

bool 
runnerFunction::Blackman_Nuttall(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    Window_Blackman_Nuttall(((std::vector<float>*)data)->data(), OFullSize, windowSize);
    return std::move(true);
}

bool 
runnerFunction::Blackman_Harris(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    Window_Blackman_harris(((std::vector<float>*)data)->data(), OFullSize, windowSize);
    return std::move(true);
}

bool 
runnerFunction::FlatTop(void* userStruct, void* data, CUI OFullSize, CUI windowSize)
{
    Window_FlatTop(((std::vector<float>*)data)->data(), OFullSize, windowSize);
    return std::move(true);
}

bool 
runnerFunction::RemoveDC(void* userStruct, void* data, CUI qtConst, CUI OFullSize, CUI windowSize)
{
    DCRemove_Common(((std::vector<float>*)data)->data(), OFullSize, windowSize);
    return std::move(true);
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
    Window_Gaussian(((std::vector<float>*)data)->data(), OFullSize, windowSize, sigma);
    return std::move(true);
}


bool 
runnerFunction::Radix6(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhpotimized6
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::Radix7(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhpotimized7
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::Radix8(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhpotimized8
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::Radix9(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhpotimized9
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::Radix10(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhpotimized10
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::Radix11(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhpotimized11
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
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
    ((std::vector<float>*)subreal)->resize(OFullSize);
    ((std::vector<float>*)subimag)->resize(OFullSize);
    CUI OHalfSize = OFullSize >> 1;
    for(unsigned int stage = 0; stage < windowRadix; ++stage)
    {
        if(stage % 2 == 0)
        {
            StockHamDITCommon
            (
                ((std::vector<float>*)real)->data(),
                ((std::vector<float>*)imag)->data(),
                ((std::vector<float>*)subreal)->data(),
                ((std::vector<float>*)subimag)->data(),
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
                ((std::vector<float>*)subreal)->data(),
                ((std::vector<float>*)subimag)->data(),
                ((std::vector<float>*)real)->data(),
                ((std::vector<float>*)imag)->data(),
                HWindowSize,
                stage,
                OHalfSize,
                windowRadix
            );
        }
        if(windowRadix % 2 != 0)
        {
            realResult = subreal;
            imagResult = subimag;
        }
    }
    return std::move(true);
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
    toHalfComplexFormat
    (
        (complex *)(((std::vector<float>*)out)->data()),
        ((std::vector<float>*)realResult)->data(),
        ((std::vector<float>*)imagResult)->data(),
        OHalfSize,
        windowRadix
    );
    return std::move(true);
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
    toPower
    (
        ((std::vector<float>*)out)->data(),
        ((std::vector<float>*)realResult)->data(),
        ((std::vector<float>*)realResult)->data(),
        OFullSize
    );
    return std::move(true);
}

