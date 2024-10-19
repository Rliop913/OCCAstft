#include <iostream>
#include "occaSample.hpp"

occaSTFT::occaSTFT(const std::string& mode, const int platform_id, const int device_id)
{
    occa::json prop;
    prop["mode"] = mode;
    prop["platform_id"] = platform_id;
    prop["device_id"] = device_id;
    prop["defines/__NEED_PI"] = "";
    // prop["verbose"] = true;
    // prop["kernel/verbose"] = true;
    // prop["kernel/compiler_flags"] = "-g";
    dev.setup(prop);
    overlap_common = dev.buildKernel("../OKL/EXPCommon.okl", "Overlap_Common", prop);
    
    Butterfly_Common = dev.buildKernel("../OKL/EXPCommon.okl", "StockHamCommon", prop);
    
    Hanning = dev.buildKernel("../OKL/EXPCommon.okl", "Window_Hanning", prop);
    Hamming = dev.buildKernel("../OKL/EXPCommon.okl", "Window_Hamming", prop);
    Blackman = dev.buildKernel("../OKL/EXPCommon.okl", "Window_Blackman", prop);
    Nuttall = dev.buildKernel("../OKL/EXPCommon.okl", "Window_Nuttall", prop);
    Blackman_Nuttall = dev.buildKernel("../OKL/EXPCommon.okl", "Window_Blackman_Nuttall", prop);
    Blackman_harris = dev.buildKernel("../OKL/EXPCommon.okl", "Window_Blackman_harris", prop);
    FlatTop = dev.buildKernel("../OKL/EXPCommon.okl", "Window_FlatTop", prop);
    Gaussian = dev.buildKernel("../OKL/EXPCommon.okl", "Window_Gaussian", prop);
    

    halfComplexFormat = dev.buildKernel("../OKL/kernel.okl", "toHalfComplexFormat", prop);
    poweredReturn = dev.buildKernel("../OKL/kernel.okl", "toPower", prop);

    TWO_POW6 = dev.buildKernel("../OKL/EXP6.okl", "Stockhoptimized6", prop);
    TWO_POW7 = dev.buildKernel("../OKL/EXP7.okl", "Stockhoptimized7", prop);
    TWO_POW8 = dev.buildKernel("../OKL/EXP8.okl", "Stockhoptimized8", prop);
    TWO_POW9 = dev.buildKernel("../OKL/EXP9.okl", "Stockhoptimized9", prop);
    TWO_POW10 = dev.buildKernel("../OKL/EXP10.okl", "Stockhoptimized10", prop);
    TWO_POW11 = dev.buildKernel("../OKL/EXP11.okl", "Stockhoptimized11", prop);
}

std::vector<float>
occaSTFT::DO(std::vector<float>& data, const CUI_ windowSizeEXP, const float overlapRatio)
{
    args.setArgs(data.size(), windowSizeEXP, overlapRatio);
    occa::memory dataIn = dev.malloc<float>(args.FullSize);
    occa::memory dataOut = dev.malloc<float>(args.OFullSize);
    occa::memory FReal = dev.malloc<float>(args.OFullSize);
    occa::memory FImag = dev.malloc<float>(args.OFullSize);
    occa::memory SReal = dev.malloc<float>(args.OFullSize);
    occa::memory SImag = dev.malloc<float>(args.OFullSize);

    dataIn.copyFrom(data.data());

    overlap_common
    ( 
        dataIn, 
        args.OFullSize, 
        args.FullSize, 
        windowSizeEXP, 
        args.OMove,
        FReal
    );
    Hanning
    (
        FReal,
        args.OFullSize,
        args.windowSize
    );
    switch (windowSizeEXP)
    {
    case 6:
        TWO_POW6
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 7:
        TWO_POW7
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 8:
        TWO_POW8
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 9:
        TWO_POW9
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 10:
        TWO_POW10
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 11:
        TWO_POW11
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    default:
        for(unsigned int stage = 0; stage < windowSizeEXP; ++stage)
        {
            if(stage % 2 == 0)
            {
                Butterfly_Common
                (
                    FReal,
                    FImag,
                    SReal,
                    SImag,
                    (args.windowSize / 2),
                    stage,
                    args.OHalfSize,
                    windowSizeEXP
                );
            }
            else
            {
                Butterfly_Common
                (
                    SReal,
                    SImag,
                    FReal,
                    FImag,
                    (args.windowSize / 2),
                    stage,
                    args.OHalfSize,
                    windowSizeEXP
                );
            }
        }
    }

    if(windowSizeEXP > 11)
    {
        if(windowSizeEXP % 2 != 0)
        {
            poweredReturn
            (
                dataOut,
                SReal,
                SImag,
                args.OFullSize
            );
        }
    }
    else
    {
        poweredReturn
        (
            dataOut,
            FReal,
            FImag,
            args.OFullSize
        );
    }

    std::vector<float> out(args.OFullSize);
    dataOut.copyTo(out.data());
    return std::move(out);
}

int main(int, char**){
    occaSTFT ostft("serial", 0, 0);
    std::vector<float> dataset(10240);
    float temp=0;
    for(auto& i : dataset)
    {
        i += temp;
        temp += 1.0 / dataset.size();
    }
    auto result = ostft.DO(dataset, 10, 0);

    for(auto i : result)
    {
        std::cout << i << ",";
    }

    getchar();
}
