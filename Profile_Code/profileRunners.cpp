#include "STFTProxy.hpp"
#include <iostream>

void
dataFiller(float* fills, unsigned int size)
{
    for(unsigned int i=0; i<size; ++i)
    {
        (*fills) = (float(i)) / (float(size)) + float(i%2);
        ++fills;
    }
}

void 
dataWait(STFTProxy& proxy)
{
    for(int i = 6; i< 21; ++i)
    {
        int radixed = 1 << i;
        int movenumber = 200000000 / radixed;
        if(movenumber > 1000)
        {
            movenumber /=1000;
        }
        else if(movenumber > 100)
        {
            movenumber /= 100;
        }
        else if(movenumber > 10)
        {
            movenumber /= 10;
        }
        
        for(unsigned long inner_itr = 1; inner_itr * radixed < 200000000; inner_itr += movenumber)
        {
            std::vector<float> data(radixed * inner_itr);
            auto dataResult = proxy.RequestSTFT(data, i, 0);
            if(!dataResult.has_value())
            {
                std::cout << "data err on return on Radix: "<< i << " datasize: " << data.size() << std::endl;
            }
            else
            {
                auto dresult = dataResult.value().get();
                if(dresult.has_value())
                {
                    std::cout << "calculate complete on RADIX: "<< i << " DATASIZE: " << data.size() <<std::endl;
                }
                else
                {
                    std::cout << "data err on calculate return on Radix: "<< i << " datasize: " << data.size() << std::endl;
                }
            }
        }
    }
}

int main()
{
    FallbackList fallbacks;
    // fallbacks.SerialFallback.push_back("./cross_gpgpu/Serial");
//    fallbacks.OpenCLFallback.push_back("./cross_gpgpu/OpenCL");
    // fallbacks.CUDAFallback.push_back("./cross_gpgpu/CUDA");
    //  fallbacks.CustomFallback.push_back("./cross_gpgpu/testRunner/testRunner.exe");
    fallbacks.CustomFallback.push_back("./cross_gpgpu/occaprofileRunner/occaRun.exe");
	// fallbacks.ServerFallback.push_back("127.0.0.1:54500");
    STFTProxy proxy
    (
        [](const ix::WebSocketErrorInfo& e){
            std::cout << e.reason <<std::endl;
	    return;
        },
        fallbacks
    );
    
    //test first
    
    dataWait(proxy);

    proxy.KillRunner(true);
    
    return 0;
}
