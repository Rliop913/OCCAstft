#include <chrono>
#include <fstream>
#include "nlohmann/json.hpp"

#include "RunnerInterface.hpp"
#include "okl_embed.hpp"
#include "clfftImpl.hpp"
#include "cufftImpl.hpp"
//include your gpgpu kernel codes.


/*
IMPLEMENTATION LISTS

1. make Genv
2. make Gcodes
3. implement BuildKernel, ActivateSTFT
4. End. try test

*/



// Genv: Structure to hold the GPGPU environment settings and resources.
struct Genv{
    // clfftImpl clf;
    cufftImpl cuf;
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    
};

// InitEnv: Initializes the GPGPU environment and kernel code structures.
// Allocates memory for 'env' (Genv) and 'kens' (Gcodes).
void
Runner::InitEnv()
{
    
    env = new Genv;
    kens = new Gcodes;
    // env->clf.init();
}

// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    //
}

void
Runner::UnInit()
{
    // env->clf.uninit();
}

/**
 * ActivateSTFT: Executes the Short-Time Fourier Transform (STFT) on the input data using GPGPU.
 * @param inData: Input signal data.
 * @param windowRadix: Radix size of the STFT window.
 * @param overlapRatio: Overlap ratio for the STFT window. 0 ~ 1, 0 means no overlap.
 * @return MAYBE_DATA: Processed data after applying STFT. if error occurs, return std::nullopt
 */

void
JsonStore
(
    unsigned int windowSize,
    unsigned int BatchSize,
    const std::string& vendor,
    unsigned int NanoSecond
)
{
    using json = nlohmann::json;
    std::ifstream dataFile("./executeResult.json");
    json data = json::parse(dataFile);
    std::string WS = std::to_string(windowSize);
    std::string DS = std::to_string(BatchSize);
    
    std::string vend = WS + vendor + DS;
    
    data[vend] = NanoSecond;
    std::ofstream storeFile("./executeResult.json");
    storeFile << std::setw(4) << data <<std::endl;

}



MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio)
{
    //default code blocks
    const unsigned int  FullSize    = inData.size();
    const int           windowSize  = 1 << windowRadix;
    const int           qtConst     = toQuot(FullSize, overlapRatio, windowSize);//number of windows
    const unsigned int  OFullSize   = qtConst * windowSize; // overlaped fullsize
    const unsigned int  OHalfSize   = OFullSize / 2;
    const unsigned int  OMove       = windowSize * (1.0f - overlapRatio);// window move distance
    //end default

    // std::cout<< "init complete"<<std::endl;
    dataSet dsets;
    dsets.FullSize = FullSize;
    dsets.OFullSize = OFullSize;
    dsets.qtConst = qtConst;
    dsets.OHalfSize= OHalfSize;
    dsets.OMove = OMove;
    dsets.overlapRatio=overlapRatio;
    dsets.windowRadix=windowRadix;
    dsets.windowSize=windowSize;
    // std::cout<< "TR:117" << std::endl;
    // clfftImpl clf;
    // clf.init();
    auto cuf_result = env->cuf.GetTime(inData, dsets);
    // clf.uninit();
    // cufftImpl cuf;
    // auto cuf_result = cuf.GetTime(inData, dsets);
    JsonStore(windowSize, qtConst, "CUFFT", cuf_result);
    // std::cout<< "CLFFT RESULT: "<< clf_result <<"nanoseconds"<<std::endl;


    std::vector<float> clout(OFullSize);
    

    return clout; // If any error occurs during STFT execution, the function returns std::nullopt.
}
