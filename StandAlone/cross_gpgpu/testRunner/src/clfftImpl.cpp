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
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    props[1] = (cl_context_properties)platform;

    context = clCreateContext(props, 1, &device, notify, NULL, &cctxt);
    cl_queue_properties pp[3];
    pp[0] = CL_QUEUE_PROPERTIES;
    pp[1] = CL_QUEUE_PROFILING_ENABLE;
    pp[2] = 0;

    CQ = clCreateCommandQueueWithProperties(context, device, pp, &cctxt);

    clfftInitSetupData(&clSetUp);
    clfftSetup(&clSetUp);
}

unsigned long long
clfftImpl::GetTime(VECF inData, const dataSet& sets)
{
    size_t clleng[1];
    clleng[0] = (1 << sets.windowSizeEXP);
    cl_int err_ret;
    float* dt = new float[sets.OFullSize * 2];
    clinput = clCreateBuffer
    (
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 2 * sets.OFullSize,
        dt,
        &err_ret
    );
    int planerr[6];
    planerr[0] =
    clfftCreateDefaultPlan(
        &planhandle, 
        context,
        CLFFT_1D,
        clleng);
    clfftSetPlanPrecision(planhandle, CLFFT_SINGLE);
    clfftSetLayout(planhandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    clfftSetResultLocation(planhandle, CLFFT_INPLACE);
    clfftSetPlanBatchSize(planhandle, sets.qtConst);
    
    clfftBakePlan(planhandle, 1, &CQ, NULL, NULL);
    cl_int eventErrcode;
    cl_event clevent = clCreateUserEvent(context, &eventErrcode);
    if(eventErrcode != CL_SUCCESS)
    {
        std::cerr<<"create event ERR, code: " << eventErrcode<<std::endl;
    }
    auto worked =   clfftEnqueueTransform(
                        planhandle,
                        CLFFT_FORWARD,
                        1,
                        &CQ,
                        0,
                        NULL,
                        &clevent,
                        &clinput,
                        &clinput,
                        NULL
                    );
    
    clFinish(CQ);
    clWaitForEvents(1, &clevent);
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
    
    size_t clstart, clend;
    
    clGetEventProfilingInfo
    (
        clevent, 
        CL_PROFILING_COMMAND_START, 
        sizeof(clstart), 
        &clstart, 
        NULL
    );
    
    clGetEventProfilingInfo
    (
        clevent, 
        CL_PROFILING_COMMAND_COMPLETE, 
        sizeof(clend), 
        &clend, 
        NULL
    );
    unsigned long long clresult = clend - clstart;


    clfftDestroyPlan(&planhandle);
    clReleaseEvent(clevent);
    clReleaseMemObject(clinput);
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