#pragma once
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <thread>
#include <functional>

#include <IXWebSocket.h>
#include <IXWebSocketServer.h>
#include "FFTStruct.hpp"

#define LOCAL_SIZE 256

using VECF = std::vector<float>;
using MAYBE_DATA = std::optional<VECF>;
using BIN = std::string;

struct Genv;
struct Gcodes;

struct cplx_t{
    float real;
    float imag;
};

inline
int 
toQuot(float fullSize, float overlapRatio, float windowSize){
    if(overlapRatio == 0.0f){
        return fullSize / windowSize + 1;
    }
    else{
        return ((fullSize ) / (windowSize * (1.0f - overlapRatio))) + 1;
    }
}

struct PreProcess{
    std::function<void(float*, const unsigned int, const unsigned int)> Hanning;
    std::function<void(float*, const unsigned int, const unsigned int)> Hamming;
    std::function<void(float*, const unsigned int, const unsigned int)> Blackman;
    std::function<void(float*, const unsigned int, const unsigned int)> Nuttall;
    std::function<void(float*, const unsigned int, const unsigned int)> Blackman_Nuttall;
    std::function<void(float*, const unsigned int, const unsigned int)> Blackman_Harris;
    std::function<void(float*, const unsigned int, const unsigned int)> FlatTop;
    std::function<void(float*, const unsigned int, const unsigned int, const float)> Gaussian;
    std::function<void(float*, const unsigned int, const unsigned int)> Remove_DC;
    
    void UseOption( const std::string& options, 
                    float* outReal, 
                    const unsigned int& OFullSize,
                    const unsigned int& windowSize)
    {
        if(options.find("--hanning_window") != std::string::npos)
        {
            Hanning(outReal, OFullSize, windowSize);
        }
        else if(options.find("--hamming_window") != std::string::npos)
        {
            Hamming(outReal, OFullSize, windowSize);
        }
        else if(options.find("--blackman_window") != std::string::npos)
        {
            Blackman(outReal, OFullSize, windowSize);
        }
        else if(options.find("--nuttall_window") != std::string::npos)
        {
            Nuttall(outReal, OFullSize, windowSize);
        }
        else if(options.find("--blackman_nuttall_window") != std::string::npos)
        {
            Blackman_Nuttall(outReal, OFullSize, windowSize);
        }
        else if(options.find("--blackman_harris_window") != std::string::npos)
        {
            Blackman_Harris(outReal, OFullSize, windowSize);
        }
        else if(options.find("--flattop_window") != std::string::npos)
        {
            FlatTop(outReal, OFullSize, windowSize);
        }
        else if(options.find("--gaussian_window=") != std::string::npos)
        {
            if(options.find("<<sigma") != std::string::npos)
            {
                auto P1 = options.find("--gaussian_window=") + 19;
                auto P2 = options.find("<<sigma");
                float sigma = -1.0f;
                std::string sigmaString = options.substr(P1, P2 - P1);
                try
                {
                    sigma = std::stof(sigmaString);
                }
                catch(const std::exception& e)
                {
                    
                }
                if(sigma > 0)
                {
                    Gaussian(outReal, OFullSize, windowSize, sigma);
                }
            }
        }
        if(options.find("--remove_dc") != std::string::npos)
        {
            Remove_DC(outReal, OFullSize, windowSize);
        }
    }
};

struct PreProcess_voidP{
    std::function<void(void*, void*, const unsigned int, const unsigned int)> Hanning;
    std::function<void(void*, void*, const unsigned int, const unsigned int)> Hamming;
    std::function<void(void*, void*, const unsigned int, const unsigned int)> Blackman;
    std::function<void(void*, void*, const unsigned int, const unsigned int)> Nuttall;
    std::function<void(void*, void*, const unsigned int, const unsigned int)> Blackman_Nuttall;
    std::function<void(void*, void*, const unsigned int, const unsigned int)> Blackman_Harris;
    std::function<void(void*, void*, const unsigned int, const unsigned int)> FlatTop;
    std::function<void(void*, void*, const unsigned int, const unsigned int, const float)> Gaussian;
    std::function<void(void*, void*, const unsigned int, const unsigned int)> Remove_DC;
    
    void UseOption( const std::string& options, 
                    void* usrPointer,
                    void* outReal, 
                    const unsigned int& OFullSize,
                    const unsigned int& windowSize)
    {
        if(options.find("--hanning_window") != std::string::npos)
        {
            Hanning(usrPointer, outReal, OFullSize, windowSize);
        }
        else if(options.find("--hamming_window") != std::string::npos)
        {
            Hamming(usrPointer, outReal, OFullSize, windowSize);
        }
        else if(options.find("--blackman_window") != std::string::npos)
        {
            Blackman(usrPointer, outReal, OFullSize, windowSize);
        }
        else if(options.find("--nuttall_window") != std::string::npos)
        {
            Nuttall(usrPointer, outReal, OFullSize, windowSize);
        }
        else if(options.find("--blackman_nuttall_window") != std::string::npos)
        {
            Blackman_Nuttall(usrPointer, outReal, OFullSize, windowSize);
        }
        else if(options.find("--blackman_harris_window") != std::string::npos)
        {
            Blackman_Harris(usrPointer, outReal, OFullSize, windowSize);
        }
        else if(options.find("--flattop_window") != std::string::npos)
        {
            FlatTop(usrPointer, outReal, OFullSize, windowSize);
        }
        else if(options.find("--gaussian_window=") != std::string::npos)
        {
            if(options.find("<<sigma") != std::string::npos)
            {
                auto P1 = options.find("--gaussian_window=") + 19;
                auto P2 = options.find("<<sigma");
                float sigma = -1.0f;
                std::string sigmaString = options.substr(P1, P2 - P1);
                try
                {
                    sigma = std::stof(sigmaString);
                }
                catch(const std::exception& e)
                {
                    
                }
                if(sigma > 0)
                {
                    Gaussian(usrPointer, outReal, OFullSize, windowSize, sigma);
                }
            }
        }
        if(options.find("--remove_dc") != std::string::npos)
        {
            Remove_DC(usrPointer, outReal, OFullSize, windowSize);
        }
    }
};
struct Runner{
private:

    void InitEnv();
    void BuildKernel();
    void UnInit();
    
    bool ServerInit(const int& pNum);//common impl
    void ServerConnect();//common impl
    MAYBE_DATA AccessData(FFTRequest& req);//common impl
    PreProcess pps;
    PreProcess_voidP vps;
    Genv *env = nullptr;
    Gcodes *kens = nullptr;
    MAYBE_SHOBJ dataInfo = std::nullopt;
public:
    [[nodiscard]]
    MAYBE_DATA
    ActivateSTFT(   VECF& inData,
                    const int& windowRadix, 
                    const float& overlapRatio,
                    const std::string& options);
    ix::WebSocketServer *server = nullptr;
    Runner(const int& portNumber);//common impl
    ~Runner();//common impl

};