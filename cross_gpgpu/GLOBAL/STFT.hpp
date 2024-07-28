#include <vector>
#include <string>
#include <optional>
#include "miniaudio.h"

#define LOCAL_SIZE 256
using VECF = std::vector<float>;
using MAYBE = std::optional<std::vector<float>>;

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


struct STFT{
private:
    void InitEnv();
    void BuildKernel();


    Genv *env;
    Gcodes *kens;

public:
    STFT();
    ~STFT();
    MAYBE
    ActivateSTFT(   VECF& inData,
                    const int& windowRadix, 
                    const float& overlapRatio);

};