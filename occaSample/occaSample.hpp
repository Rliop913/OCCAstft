#pragma once

#include <occa.hpp>
using UI_ = unsigned int;
using CUI_ = const unsigned int;
struct STFTArgs{
    UI_  FullSize;
    UI_  windowSize;
    UI_  qtConst;
    UI_  OFullSize;
    UI_  OHalfSize;
    UI_  OMove;
    void setArgs(CUI_ dataSize, CUI_ windowRadix, const float overlapRatio)
    {
        FullSize = dataSize;
        windowSize =  1 << windowRadix;
        if(overlapRatio == 0.0f){
            qtConst = FullSize / windowSize + 1;
        }
        else{
            qtConst = ((FullSize) / (windowSize * (1.0f - overlapRatio))) + 1;
        }
        OFullSize = qtConst * windowSize;
        OHalfSize = OFullSize >> 1;
        OMove = windowSize * (1.0f - overlapRatio);
    }
};


struct occaSTFT{
private:
    STFTArgs args;
    occa::device dev;


    occa::kernel overlap_common;
    
    occa::kernel Hanning;
    occa::kernel Hamming;
    occa::kernel Blackman;
    occa::kernel Nuttall;
    occa::kernel Blackman_Nuttall;
    occa::kernel Blackman_harris;
    occa::kernel FlatTop;
    occa::kernel Gaussian;
    
    occa::kernel Butterfly_Common;
    occa::kernel R6;
    occa::kernel R7;
    occa::kernel R8;
    occa::kernel R9;
    occa::kernel R10;
    occa::kernel R11;
    
    occa::kernel halfComplexFormat;
    occa::kernel poweredReturn;
public:
    std::vector<float> DO(std::vector<float>& data, const CUI_ windowRadix, const float overlapRatio);
    occaSTFT(const std::string& mode, const int platform_id, const int device_id);
    ~occaSTFT(){;}
};