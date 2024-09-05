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


int main()
{
    FallbackList fallbacks;
    fallbacks.CustomFallback.push_back("");
    STFTProxy proxy
    (
        [](const ix::WebSocketErrorInfo& e){
            return;
        },
        fallbacks
    );
    
    //test first
    std::vector<float> D5(10000);
    std::vector<MAYBE_FUTURE_DATA> D5Ret;
    dataFiller(D5.data(), D5.size());
    D5Ret.push_back(proxy.RequestSTFT(D5, 10, 0.0));
    D5Ret.push_back(proxy.RequestSTFT(D5, 11, 0.0));
    D5Ret.push_back(proxy.RequestSTFT(D5, 12, 0.0));
    D5Ret.push_back(proxy.RequestSTFT(D5, 13, 0.0));
    D5Ret.push_back(proxy.RequestSTFT(D5, 14, 0.0));
    D5Ret.push_back(proxy.RequestSTFT(D5, 15, 0.0));
    for(auto& i : D5Ret)
    {
        if(i.has_value())
        {
            auto returnVal = i.value().get();
            if(!returnVal.has_value())
            {
                std::cout << "return Error on D5 DataSet!!!" << std::endl;
            }
        }
    }
    std::cout << "D5 Complete" << std::endl;
    std::vector<float> D6(100000);
    std::vector<MAYBE_FUTURE_DATA> D6Ret;
    dataFiller(D6.data(), D6.size());
    D6Ret.push_back(proxy.RequestSTFT(D6, 10, 0.0));
    D6Ret.push_back(proxy.RequestSTFT(D6, 11, 0.0));
    D6Ret.push_back(proxy.RequestSTFT(D6, 12, 0.0));
    D6Ret.push_back(proxy.RequestSTFT(D6, 13, 0.0));
    D6Ret.push_back(proxy.RequestSTFT(D6, 14, 0.0));
    D6Ret.push_back(proxy.RequestSTFT(D6, 15, 0.0));
    for(auto& i : D6Ret)
    {
        if(i.has_value())
        {
            auto returnVal = i.value().get();
            if(!returnVal.has_value())
            {
                std::cout << "return Error on D6 DataSet!!!" << std::endl;
            }
        }
    }
    std::cout << "D6 Complete" << std::endl;
    std::vector<float> D7(1000000);
    std::vector<MAYBE_FUTURE_DATA> D7Ret;
    dataFiller(D7.data(), D7.size());
    D7Ret.push_back(proxy.RequestSTFT(D7, 10, 0.0));
    D7Ret.push_back(proxy.RequestSTFT(D7, 11, 0.0));
    D7Ret.push_back(proxy.RequestSTFT(D7, 12, 0.0));
    D7Ret.push_back(proxy.RequestSTFT(D7, 13, 0.0));
    D7Ret.push_back(proxy.RequestSTFT(D7, 14, 0.0));
    D7Ret.push_back(proxy.RequestSTFT(D7, 15, 0.0));
    for(auto& i : D7Ret)
    {
        if(i.has_value())
        {
            auto returnVal = i.value().get();
            if(!returnVal.has_value())
            {
                std::cout << "return Error on D7 DataSet!!!" << std::endl;
            }
        }
    }
    std::cout << "D7 Complete" << std::endl;
    std::vector<float> D8(10000000);
    std::vector<MAYBE_FUTURE_DATA> D8Ret;
    dataFiller(D8.data(), D8.size());
    D8Ret.push_back(proxy.RequestSTFT(D8, 10, 0.0));
    D8Ret.push_back(proxy.RequestSTFT(D8, 11, 0.0));
    D8Ret.push_back(proxy.RequestSTFT(D8, 12, 0.0));
    D8Ret.push_back(proxy.RequestSTFT(D8, 13, 0.0));
    D8Ret.push_back(proxy.RequestSTFT(D8, 14, 0.0));
    D8Ret.push_back(proxy.RequestSTFT(D8, 15, 0.0));
    for(auto& i : D8Ret)
    {
        if(i.has_value())
        {
            auto returnVal = i.value().get();
            if(!returnVal.has_value())
            {
                std::cout << "return Error on D8 DataSet!!!" << std::endl;
            }
        }
    }
    std::cout << "D8 Complete" << std::endl;

    std::vector<float> D9(100000000);
    std::vector<MAYBE_FUTURE_DATA> D9Ret;
    dataFiller(D9.data(), D9.size());
    D9Ret.push_back(proxy.RequestSTFT(D9, 10, 0.0));
    D9Ret.push_back(proxy.RequestSTFT(D9, 11, 0.0));
    D9Ret.push_back(proxy.RequestSTFT(D9, 12, 0.0));
    D9Ret.push_back(proxy.RequestSTFT(D9, 13, 0.0));
    D9Ret.push_back(proxy.RequestSTFT(D9, 14, 0.0));
    D9Ret.push_back(proxy.RequestSTFT(D9, 15, 0.0));
    for(auto& i : D9Ret)
    {
        if(i.has_value())
        {
            auto returnVal = i.value().get();
            if(!returnVal.has_value())
            {
                std::cout << "return Error on D9 DataSet!!!" << std::endl;
            }
        }
    }
    std::cout << "D9 Complete" << std::endl;

    
    return 0;
}