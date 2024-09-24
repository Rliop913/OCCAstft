#include "cufftImpl.hpp"

void
cufftImpl::init()
{

}

unsigned long long
cufftImpl::GetTime(VECF inData, const dataSet& sets)
{
    cufftHandle plan;
    cufftComplex *cplxdata;

    std::cout << "cumalloc err code: " << cudaMalloc((void**)&cplxdata, sizeof(cufftComplex) * sets.OFullSize)<<"  --  "<< sizeof(cufftComplex) <<std::endl;

    if(cufftPlan1d(&plan, sets.windowSize, CUFFT_C2C, sets.qtConst) != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT ERR, plan create Fail" <<cufftPlan1d(&plan, sets.windowSize, CUFFT_C2C, sets.qtConst) <<std::endl;
        return 7;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    if(cufftExecC2C(plan, cplxdata, cplxdata, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cout<< "CUFFT err, exec forward Fail" <<std::endl;
        return 8;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float millisec = 0;
    cudaEventElapsedTime(&millisec, start, stop);
    unsigned long long nano = (millisec * 1000000);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cufftDestroy(plan);
    cudaFree(cplxdata);
    return nano;
}

void 
cufftImpl::uninit()
{

}