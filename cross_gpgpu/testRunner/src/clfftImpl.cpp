#include "clfftImpl.hpp"

void
notify (const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    std::cout<< errinfo <<std::endl;
    
}

void
clfftImpl::init()
{
    int cctxt;
    cctxt = clGetPlatformIDs(1, &platform, NULL);
    cctxt = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, notify, NULL, &cctxt);
    cl_queue_properties pp[1];
    pp[0] = CL_QUEUE_PROFILING_ENABLE;
    CQ = clCreateCommandQueueWithProperties(context, device, pp, &cctxt);
   
    cctxt = clfftInitSetupData(&clSetUp);
    cctxt = clfftSetup(&clSetUp);
    std::cout<<"clfft init complete"<<std::endl;
}

unsigned long long
clfftImpl::GetTime(VECF inData, const dataSet& sets)
{
    size_t clleng[1];
    clleng[0] = (1 << sets.windowRadix);
    cl_int err_ret;
    float* dt = new float[sets.OFullSize];
    memcpy(dt, inData.data(), sizeof(float) * inData.size());
    std::cout<< "point 54" <<std::endl;
    clinput = clCreateBuffer
    (
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * sets.OFullSize,
        dt,
        &err_ret
    );
    std::cout<< "point 54" <<std::endl;
    int planerr[6];
    planerr[0] =
    clfftCreateDefaultPlan(
        &planhandle, 
        context,
        CLFFT_1D,
        clleng);
        std::cout<< "point 54" <<std::endl;
    planerr[1] =
    clfftSetPlanPrecision(planhandle, CLFFT_SINGLE);
    planerr[2] =
    clfftSetLayout(planhandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    planerr[3] =
    clfftSetResultLocation(planhandle, CLFFT_INPLACE);
    planerr[4] =
    clfftSetPlanBatchSize(planhandle, sets.qtConst);
    std::cout<< "point 54" <<std::endl;
    planerr[5] =
    clfftBakePlan(planhandle, 1, &CQ, NULL, NULL);
    
    cl_event clevent = clCreateUserEvent(context, NULL);
    std::cout<< "point 54" <<std::endl;
    auto worked =   clfftEnqueueTransform(
                        planhandle,
                        CLFFT_FORWARD,
                        1,
                        &CQ,
                        0,
                        NULL,
                        &clevent,
                        &clinput,
                        NULL,
                        NULL
                    );
    clFinish(CQ);
    if(worked != CL_SUCCESS)
    {
        std::cerr<< "it didn't worked. do not trust clfft timer" <<std::endl;
    }
    auto read_result =
    clEnqueueReadBuffer
    (
        CQ,
        clinput,
        CL_TRUE,
        0,
        sizeof(float) * sets.OFullSize,
        dt,
        0,
        NULL,
        NULL
    );
    
    cl_ulong clstart, clend;
    clGetEventProfilingInfo
    (
        clevent, 
        CL_PROFILING_COMMAND_START, 
        sizeof(clstart), 
        &clstart, 
        NULL
    );
	std::cout << "TR:171" <<std::endl;
    clGetEventProfilingInfo
    (
        clevent, 
        CL_PROFILING_COMMAND_END, 
        sizeof(clend), 
        &clend, 
        NULL
    );
    cl_ulong clresult = clend - clstart;


    std::cout << clfftDestroyPlan(&planhandle) << std::endl;
    std::cout << clReleaseEvent(clevent) << std::endl;
    std::cout << clReleaseMemObject(clinput) << std::endl;
    delete[] dt;
    return clresult;
}

void
clfftImpl::uninit()
{
    clfftTeardown();
    clReleaseDevice(device);
    clReleaseCommandQueue(CQ);
    clReleaseContext(context);
}