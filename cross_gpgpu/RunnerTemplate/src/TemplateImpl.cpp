#include "RunnerInterface.hpp"
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
    //
};

// Gcodes: Structure to manage and store GPGPU kernel codes.
struct Gcodes{
    //
};

// InitEnv: Initializes the GPGPU environment and kernel code structures.
// Allocates memory for 'env' (Genv) and 'kens' (Gcodes).
void
Runner::InitEnv()
{
    env = new Genv;
    kens = new Gcodes;
}

// BuildKernel: Compiles or prepares the GPGPU kernel for execution.
void
Runner::BuildKernel()
{
    //
}


/**
 * ActivateSTFT: Executes the Short-Time Fourier Transform (STFT) on the input data using GPGPU.
 * @param inData: Input signal data.
 * @param windowRadix: Radix size of the STFT window.
 * @param overlapRatio: Overlap ratio for the STFT window. 0 ~ 1, 0 means no overlap.
 * @return MAYBE_DATA: Processed data after applying STFT. if error occurs, return std::nullopt
 */

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

    // memory allocation samples
    auto tempMem = new complex  [OFullSize]();
    auto qtBuffer= new float    [qtConst]();
    std::vector<float> outMem(OHalfSize);

    // Implement your GPGPU kernel execution code here.

    

    delete[] tempMem;
    delete[] qtBuffer;

    return std::move(outMem); // If any error occurs during STFT execution, the function returns std::nullopt.
}