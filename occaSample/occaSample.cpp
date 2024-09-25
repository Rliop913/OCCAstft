#include <iostream>
#include "occaSample.hpp"

occaSTFT::occaSTFT(const std::string& mode, const int platform_id, const int device_id)
{
    occa::json prop;
    prop["mode"] = mode;
    prop["platform_id"] = platform_id;
    prop["device_id"] = device_id;
    prop["defines/__NEED_PI"] = "";
    dev.setup(prop);
    overlap_common = dev.buildKernel("../OKL/RadixCommon.okl", "Overlap_Common", prop);
    
    Butterfly_Common = dev.buildKernel("../OKL/RadixCommon.okl", "StockHamDITCommon", prop);
    
    Hanning = dev.buildKernel("../OKL/RadixCommon.okl", "Window_Hanning", prop);
    Hamming = dev.buildKernel("../OKL/RadixCommon.okl", "Window_Hamming", prop);
    Blackman = dev.buildKernel("../OKL/RadixCommon.okl", "Window_Blackman", prop);
    Nuttall = dev.buildKernel("../OKL/RadixCommon.okl", "Window_Nuttall", prop);
    Blackman_Nuttall = dev.buildKernel("../OKL/RadixCommon.okl", "Window_Blackman_Nuttall", prop);
    Blackman_harris = dev.buildKernel("../OKL/RadixCommon.okl", "Window_Blackman_harris", prop);
    FlatTop = dev.buildKernel("../OKL/RadixCommon.okl", "Window_FlatTop", prop);
    Gaussian = dev.buildKernel("../OKL/RadixCommon.okl", "Window_Gaussian", prop);
    

    halfComplexFormat = dev.buildKernel("../OKL/kernel.okl", "toHalfComplexFormat", prop);
    poweredReturn = dev.buildKernel("../OKL/kernel.okl", "toPower", prop);

    R6 = dev.buildKernel("../OKL/Radix6.okl", "Stockhpotimized6", prop);
    R7 = dev.buildKernel("../OKL/Radix7.okl", "Stockhpotimized7", prop);
    R8 = dev.buildKernel("../OKL/Radix8.okl", "Stockhpotimized8", prop);
    R9 = dev.buildKernel("../OKL/Radix9.okl", "Stockhpotimized9", prop);
    R10 = dev.buildKernel("../OKL/Radix10.okl", "Stockhpotimized10", prop);
    R11 = dev.buildKernel("../OKL/Radix11.okl", "Stockhpotimized11", prop);
}

std::vector<float>
occaSTFT::DO(std::vector<float>& data, const CUI_ windowRadix, const float overlapRatio)
{
    args.setArgs(data.size(), windowRadix, overlapRatio);
    occa::memory dataIn = dev.malloc<float>(args.FullSize);
    occa::memory dataOut = dev.malloc<float>(args.OFullSize);
    occa::memory FReal = dev.malloc<float>(args.OFullSize);
    occa::memory FImag = dev.malloc<float>(args.OFullSize);
    occa::memory SReal = dev.malloc<float>(args.OFullSize);
    occa::memory SImag = dev.malloc<float>(args.OFullSize);

    dataIn.copyFrom(data.data());

    overlap_common\
    ( 
        dataIn, 
        args.OFullSize, 
        args.FullSize, 
        windowRadix, 
        args.OMove,
        FReal
    );
    Hanning
    (
        FReal,
        args.OFullSize,
        args.windowSize
    );
    switch (windowRadix)
    {
    case 6:
        R6
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 7:
        R7
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 8:
        R8
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 9:
        R9
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 10:
        R10
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    case 11:
        R11
        (
            FReal,
            FImag,
            args.OHalfSize
        );
        break;
    default:
        for(unsigned int stage = 0; stage < windowRadix; ++stage)
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
                    windowRadix
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
                    windowRadix
                );
            }
        }
    }

    if(windowRadix > 11)
    {
        if(windowRadix % 2 != 0)
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
    // occa::device dev;
    // occa::json prop = {{"mode", "serial"}, {"platform_id", 0}, {"device_id", 0}};
    // prop["verbose"] = true;
    // prop["kernel/verbose"] = true;
    // prop["kernel/compiler_flags"] = "-g";
    // dev.setup(prop);

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
