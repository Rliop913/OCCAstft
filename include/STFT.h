#pragma once
#include <occa.hpp>
#include "okl_embedded.h"
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <unordered_map>
const double PI = 3.141592653589793238460;

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// 재귀적 IFFT 함수

struct occacplx{
    float real;
    float imag;
};

constexpr 
int 
toQuot(float fullSize, float overlapRatio, float windowSize){
    if(overlapRatio == 0.0f){
        return fullSize / windowSize + 1;
    }
    else{
        return ((fullSize ) / (windowSize * (1.0f - overlapRatio))) + 1;
    }
}
#define CONSTRL const std::string
#define CONSTRR const std::string&

using namespace occa;
class Stft{
private:
    device dev;
    json prop;
    
public:
    Stft(CONSTRL mode);
    template<typename T>
    memory makeMem(const unsigned int& data_length){
        return dev.malloc<T>(data_length);
    }
    
    memory makeMem(const unsigned int& data_length, occa::dtype_t& type){
        return dev.malloc(data_length, type);
    }
    
    void addNewKernel(CONSTRL kernel_path, CONSTRL kernel_name);
    std::unordered_map<std::string, kernel> kern;
    ~Stft(){;}
    void do_stft();
};
