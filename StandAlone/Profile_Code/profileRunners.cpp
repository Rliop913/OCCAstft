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
    for(int i = 8; i< 21; ++i)
    {
        int exped = 1 << i;
        int movenumber = 2000000 / exped;
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
        
        for(unsigned long inner_itr = 1; inner_itr * exped < 200000000; inner_itr += movenumber)
        {
            std::vector<float> data(exped * inner_itr);
            auto dataResult = proxy.RequestSTFT(data, i, 0);
            if(!dataResult.has_value())
            {
                std::cout << "data err on return on EXP: "<< i << " datasize: " << data.size() << std::endl;
            }
            else
            {
                auto dresult = dataResult.value().get();
                if(dresult.has_value())
                {
                    std::cout << "calculate complete on EXP: "<< i << " DATASIZE: " << data.size() <<std::endl;
                }
                else
                {
                    std::cout << "data err on calculate return on EXP: "<< i << " datasize: " << data.size() << std::endl;
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
    // fallbacks.CustomFallback.push_back("./cross_gpgpu/occaprofileRunner/occaRun.exe");
	fallbacks.ServerFallback.push_back("127.0.0.1:54500");
    STFTProxy proxy
    (
        [](const ix::WebSocketErrorInfo& e){
            std::cout << e.reason <<std::endl;
	    return;
        },
        fallbacks
    );
    
    //test first
    
    // dataWait(proxy);
    std::vector<float> dataTemp(1000000);
    
    for(int i=0; i< dataTemp.size(); ++i)
    {
        dataTemp[i] = float(i) / float(dataTemp.size());
    }
    auto returned = proxy.RequestSTFT(dataTemp, 14, 0.0);
    if(returned.has_value())
    {
        auto ret = returned.value().get();
        if(!ret.has_value())
        {
            std::cout << "ERR returned" <<std::endl;
        }
        for(auto i : ret.value())
        {
            std::cout << i << ", ";
        }
    }
    else{
        std::cout << "something happened"<<std::endl;
    }

    getchar();

    proxy.KillRunner(true);
    
    return 0;
}
