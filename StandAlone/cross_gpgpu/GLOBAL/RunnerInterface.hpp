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
namespace runnerFunction {

    bool Overlap            (void* userStruct, void* origin, CUI OFullSize, CUI FullSize, CUI windowSizeEXP, CUI OMove, void* Realout);
    bool Hanning            (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
    bool Hamming            (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
    bool Blackman           (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
    bool Nuttall            (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
    bool Blackman_Nuttall   (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
    bool Blackman_Harris    (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
    bool FlatTop            (void* userStruct, void* data, CUI OFullSize, CUI windowSize);
    bool Gaussian           (void* userStruct, void* data, CUI OFullSize, CUI windowSize, const float sigma);

    bool RemoveDC           (void* userStruct, void* data, CUI qtConst, CUI OFullSize, CUI windowSize);

    bool EXP6             (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
    bool EXP7             (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
    bool EXP8             (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
    bool EXP9             (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
    bool EXP10            (void* userStruct, void* Real, void* Imag, CUI OHalfSize);
    bool EXP11            (void* userStruct, void* Real, void* Imag, CUI OHalfSize);

    bool EXPC             (   void*   userStruct,
                                void*   real, 
                                void*   imag,
                                void*   subreal,
                                void*   subimag,
                                void*   out,
                                CUI     HWindowSize,
                                CUI     windowSizeEXP,
                                CUI     OFullSize,
                                void*   realResult,
                                void*   imagResult
                            );

    bool HalfComplex        (   void*   userStruct, 
                                void*   out, 
                                void*   realResult, 
                                void*   imagResult, 
                                CUI     OHalfSize, 
                                CUI     windowSizeEXP
                            );

    bool ToPower            (   void* userStruct, 
                                void* out, 
                                void* realResult, 
                                void* imagResult, 
                                CUI OFullSize
                            );

    std::string
    Default_Pipeline(
        void* userStruct,
        void* origin,
        void* real,
        void* imag,
        void* subreal,
        void* subimag,
        void* out,
        CUI   FullSize,
        CUI   windowSize,
        CUI   qtConst,
        CUI   OFullSize,
        CUI   OHalfSize,
        CUI   OMove,
        const std::string&  options,
        const int           windowSizeEXP,
        const float         overlapRatio);

};



struct Runner{
private:

    void InitEnv();
    void BuildKernel();
    void UnInit();
    
    bool ServerInit(const int& pNum);//common impl
    void ServerConnect();//common impl
    MAYBE_DATA AccessData(FFTRequest& req);//common impl
    Genv *env = nullptr;
    Gcodes *kens = nullptr;
    MAYBE_SHOBJ dataInfo = std::nullopt;
public:
    [[nodiscard]]
    MAYBE_DATA
    ActivateSTFT(   VECF& inData,
                    const int& windowSizeEXP, 
                    const float& overlapRatio,
                    const std::string& options);
    ix::WebSocketServer *server = nullptr;
    Runner(const int& portNumber);//common impl
    ~Runner();//common impl

};