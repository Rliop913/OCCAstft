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

extern void Stockhoptimized6        (float*, float*, CUI&);
extern void Stockhoptimized7        (float*, float*, CUI&);
extern void Stockhoptimized8        (float*, float*, CUI&);
extern void Stockhoptimized9        (float*, float*, CUI&);
extern void Stockhoptimized10       (float*, float*, CUI&);
extern void Stockhoptimized11       (float*, float*, CUI&);
extern void StockHamCommon          (float*,
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
    CUI windowSizeEXP, 
    CUI OMove, 
    void* Realout
    )
{
    Overlap_Common
    (
        ((std::vector<float>*)origin)->data(),
        OFullSize,
        FullSize,
        windowSizeEXP,
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
runnerFunction::EXP6(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhoptimized6
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::EXP7(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhoptimized7
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::EXP8(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhoptimized8
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::EXP9(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhoptimized9
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::EXP10(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhoptimized10
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
}

bool 
runnerFunction::EXP11(void* userStruct, void* Real, void* Imag, CUI OHalfSize)
{
    Stockhoptimized11
    (
        ((std::vector<float>*)Real)->data(),
        ((std::vector<float>*)Imag)->data(),
        OHalfSize
    );
    return std::move(true);
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
    ((std::vector<float>*)subreal)->resize(OFullSize);
    ((std::vector<float>*)subimag)->resize(OFullSize);
    CUI OHalfSize = OFullSize >> 1;
    for(unsigned int stage = 0; stage < windowSizeEXP; ++stage)
    {
        if(stage % 2 == 0)
        {
            StockHamCommon
            (
                ((std::vector<float>*)real)->data(),
                ((std::vector<float>*)imag)->data(),
                ((std::vector<float>*)subreal)->data(),
                ((std::vector<float>*)subimag)->data(),
                HWindowSize,
                stage,
                OHalfSize,
                windowSizeEXP
            );
        }
        else
        {
            StockHamCommon
            (
                ((std::vector<float>*)subreal)->data(),
                ((std::vector<float>*)subimag)->data(),
                ((std::vector<float>*)real)->data(),
                ((std::vector<float>*)imag)->data(),
                HWindowSize,
                stage,
                OHalfSize,
                windowSizeEXP
            );
        }
        if(windowSizeEXP % 2 != 0)
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
    CUI     windowSizeEXP
    )
{
    toHalfComplexFormat
    (
        (complex *)(((std::vector<float>*)out)->data()),
        ((std::vector<float>*)realResult)->data(),
        ((std::vector<float>*)imagResult)->data(),
        OHalfSize,
        windowSizeEXP
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

