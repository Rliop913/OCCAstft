#include <vector>
#include <string>
#include <optional>
#include <IXWebSocket.h>
#include <IXWebSocketServer.h>
#include "FFTStruct.hpp"
#include "miniaudio.h"

#define FIXED_PORT 52437
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
    
    
    void ServerInit();//common impl
    void ServerConnect();//common impl

    MAYBE_DATA
    ActivateSTFT(   VECF& inData,
                    const int& windowRadix, 
                    const float& overlapRatio);
    FFTRequest Activate(const BIN& bindata);
    Genv *env = nullptr;
    Gcodes *kens = nullptr;

public:
    ix::WebSocketServer *server = nullptr;
    Runner();//common impl
    ~Runner();//common impl

};