#pragma once
#include <iostream>
#include "RunnerInterface.hpp"
struct dataSet{
    int windowRadix;
    float overlapRatio;
    unsigned int FullSize;
    int windowSize;
    int qtConst;
    unsigned int OFullSize;
    unsigned int OHalfSize;
    unsigned int OMove;
};

class calculateRAII{
public:
    virtual void init() = 0;
    virtual unsigned long long GetTime(VECF inData, const dataSet& sets) = 0;
    virtual void uninit() = 0;
    calculateRAII(){}
    ~calculateRAII(){}
};