#include <chrono>
#include <fstream>
#include "nlohmann/json.hpp"

#include "RunnerInterface.hpp"
#include "okl_embed.hpp"
#include <CL/cl.h>
#include <clFFT.h>
//include your gpgpu kernel codes.


/*
IMPLEMENTATION LISTS

1. make Genv
2. make Gcodes
3. implement BuildKernel, ActivateSTFT
4. End. try test

*/



// Genv: Structure to hold the GPGPU environment settings and resources.
struct Genv{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue CQ;
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    clfftDim dim = CLFFT_1D;
};

// InitEnv: Initializes the GPGPU environment and kernel code structures.
// Allocates memory for 'env' (Genv) and 'kens' (Gcodes).
void
Runner::InitEnv()
{
    env = new Genv;
    kens = new Gcodes;

    clGetPlatformIDs(1, &(env->platform), NULL);
    clGetDeviceIDs(env->platform, CL_DEVICE_TYPE_GPU, 1, &(env->device), NULL);
    env->context = clCreateContext(NULL, 1, &(env->device), NULL, NULL, NULL);
    env->CQ = clCreateCommandQueueWithProperties(env->context, env->device, 0, NULL);
}

// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    //
}

void
Runner::UnInit()
{
    clfftTeardown();
    clReleaseCommandQueue(env->CQ);
    clReleaseContext(env->context);
}

/**
 * ActivateSTFT: Executes the Short-Time Fourier Transform (STFT) on the input data using GPGPU.
 * @param inData: Input signal data.
 * @param windowRadix: Radix size of the STFT window.
 * @param overlapRatio: Overlap ratio for the STFT window. 0 ~ 1, 0 means no overlap.
 * @return MAYBE_DATA: Processed data after applying STFT. if error occurs, return std::nullopt
 */

void
JsonStore
(
    unsigned int windowSize,
    unsigned int DataSize,
    unsigned int cufftNanoSecond,
    unsigned int clfftNanoSecond,
    unsigned int occafftNanoSecond
)
{
    using json = nlohmann::json;
    std::ifstream dataFile("./executeResult.json");
    json data = json::parse(dataFile);
    std::string WS = std::to_string(windowSize);
    std::string DS = std::to_string(DataSize);
    std::string CUK = WS + "CUFFT" + DS;
    std::string CLK = WS + "CLFFT" + DS;
    std::string OCK = WS + "OCCAFFT" + DS;
    data[CUK] = cufftNanoSecond;
    data[CLK] = clfftNanoSecond;
    data[OCK] = occafftNanoSecond;
    std::ofstream storeFile("./executeResult.json");
    storeFile << std::setw(4) << data <<std::endl;

}



MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio)
{
    //default code blocks
    const unsigned int  FullSize    = inData.size();
    const int           windowSize  = 1 << windowRadix;
    const int           qtConst     = toQuot(FullSize, overlapRatio, windowSize);//number of windows
    const unsigned int  OFullSize   = qtConst * windowSize; // overlaped fullsize
    const unsigned int  OHalfSize   = OFullSize / 2;
    const unsigned int  OMove       = windowSize * (1.0f - overlapRatio);// window move distance
    //end default
    std::cout<< "TR:115"<<std::endl;
    size_t clleng[1];
    clleng[0] = (1 << windowRadix);
    clfftSetupData clSetUp;
    clfftInitSetupData(&clSetUp);
    clfftSetup(&clSetUp);

    clfftPlanHandle planhandle;
    clfftCreateDefaultPlan(
        &planhandle, 
        env->context,
        kens->dim,
        clleng);
    clfftSetPlanPrecision(planhandle, CLFFT_SINGLE);
    clfftSetLayout(planhandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    clfftSetResultLocation(planhandle, CLFFT_INPLACE);
std::cout<< "TR:131"<<std::endl;
    std::cout<<clfftSetPlanBatchSize(planhandle, qtConst)<<std::endl;
std::cout<< "TR:133"<<std::endl;
    cl_mem clinput = clCreateBuffer
    (
        env->context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * OFullSize,
        inData.data(),
        NULL
    );
    std::cout<< "TR:142"<<std::endl;
    std::cout<<clfftBakePlan(planhandle, 1, &(env->CQ), NULL, NULL)<<std::endl;

    cl_event clevent = clCreateUserEvent(env->context, NULL);
std::cout<< "TR:146"<<std::endl;
    clfftEnqueueTransform(
        planhandle,
        CLFFT_FORWARD,
        1,
        &(env->CQ),
        0,
        NULL,
        &clevent,
        &clinput,
        NULL,
        NULL
    );

    clFinish(env->CQ);
    std::cout<< "TR:161"<<std::endl;
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


    std::vector<float> clout(OFullSize);
    clEnqueueReadBuffer
    (
        env->CQ, 
        clinput, 
        CL_TRUE,
        0,
        sizeof(float) * OFullSize,
        clout.data(),
        0,
        NULL,
        NULL
    );
    clReleaseEvent(clevent);
    std::cout<< "TR:196"<<std::endl;
    clfftDestroyPlan(&planhandle);
    clReleaseMemObject(clinput);

    return clout; // If any error occurs during STFT execution, the function returns std::nullopt.
}
