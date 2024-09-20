#pragma once
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <thread>

#include <IXWebSocket.h>
#include <IXWebSocketServer.h>
#include "FFTStruct.hpp"


using VECF = std::vector<float>;
using MAYBE_DATA = std::optional<VECF>;
using BIN = std::string;
using CUI = const unsigned int;
using UI = unsigned int;
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

bool Hanning            (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
bool Hamming            (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
bool Blackman           (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
bool Nuttall            (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
bool Blackman_Nuttall   (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
bool Blackman_Harris    (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
bool FlatTop            (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
bool RemoveDC           (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
bool Gaussian           (void* userStruct, void* data, CUI OFullSize, CUI windowSize, const float sigma);

bool Radix6             (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
bool Radix7             (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
bool Radix8             (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
bool Radix9             (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
bool Radix10            (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
bool Radix11            (void* userStruct, void* Real, void* Imag, CUI OHalfSize);

bool RadixC             (   void*   userStruct,
                            void*   real, 
                            void*   imag,
                            void*   out,
                            CUI&&   HWindowSize,
                            CUI     windowRadix,
                            CUI     OFullSize,
                            void*   realResult,
                            void*   imagResult
                        );

bool HalfComplex        (   void*   userStruct, 
                            void*   out, 
                            void*   realResult, 
                            void*   imagResult, 
                            CUI     OHalfSize, 
                            CUI     windowRadix
                        );

bool ToPower            (   void* userStruct, 
                            void* out, 
                            void* realResult, 
                            void* imagResult, 
                            CUI OFullSize
                        );

std::string&& 
Default_Pipeline(
    void* userStruct, 
    void* real,
    void* imag,
    CUI&& FullSize,
    CUI&& windowSize,
    CUI&& qtConst,
    CUI&& OFullSize,
    CUI&& OHalfSize,
    CUI&& OMove,
    const std::string&  options,
    const int           windowRadix,
    const float         overlapRatio);




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