#pragma once
#include "template.hpp"
#include <cufft.h>
#include <cuda_runtime.h>
class cufftImpl : calculateRAII{
private:

public:
    virtual void init();
    virtual unsigned long long GetTime(VECF inData, const dataSet& sets);
    virtual void uninit();
    cufftImpl(){}
    ~cufftImpl(){}
};