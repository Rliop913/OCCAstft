#pragma once


#include "template.hpp"
#include <CL/cl.h>
#include <clFFT.h>


class clfftImpl : calculateRAII{
private:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue CQ;

    clfftSetupData clSetUp;
    clfftPlanHandle planhandle;

    cl_mem clinput;
public:
    virtual void init();
    virtual unsigned long long GetTime(VECF inData, const dataSet& sets);
    virtual void uninit();
    clfftImpl(){}
    ~clfftImpl(){}
};