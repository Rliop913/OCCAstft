#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <thread>

#include <IXWebSocket.h>
#include <IXWebSocketServer.h>
#include "FFTStruct.hpp"
#include "miniaudio.h"

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
                    const int& windowRadix, 
                    const float& overlapRatio);
    ix::WebSocketServer *server = nullptr;
    Runner(const int& portNumber);//common impl
    ~Runner();//common impl

};