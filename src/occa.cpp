#include <iostream>

#include <occa.hpp>
#include "miniaudio.h"
// #include "okl_embedded.h"
#include "STFT.h"
#include <assert.h>
#define READFRAME 1024000

ComplexVector ifft(const ComplexVector& X) {
    int N = X.size();
    if (N <= 1) {
        return X;
    }

    // 입력 크기가 2의 거듭제곱인지 확인
    if ((N & (N - 1)) != 0) {
        throw std::invalid_argument("Input size must be a power of 2");
    }

    ComplexVector X_even(N/2), X_odd(N/2);
    for (int i = 0; i < N/2; i++) {
        X_even[i] = X[2*i];
        X_odd[i] = X[2*i + 1];
    }

    ComplexVector even_ifft = ifft(X_even);
    ComplexVector odd_ifft = ifft(X_odd);

    ComplexVector result(N);
    for (int k = 0; k < N/2; k++) {
        Complex t = std::polar(1.0, 2 * PI * k / N) * odd_ifft[k];
        result[k] = even_ifft[k] + t;
        result[k + N/2] = even_ifft[k] - t;
    }

    

    return result;
}

unsigned int counter =0;
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    float* data = (float*)pDevice->pUserData;
    for(int i=0;i<frameCount;++i)
    {
        if(counter < READFRAME)
        {
            ((float*)pOutput)[i]=data[counter++];

        }
    }
}

int main(int, char**){
    occa::device dev;
    occa::json prop = {{"mode", "serial"}, {"platform_id", 0}, {"device_id", 0}};
    prop["verbose"] = true;
    prop["kernel/verbose"] = true;
    prop["kernel/compiler_flags"] = "-g";
    dev.setup(prop);
    
    ma_decoder_config decconf = ma_decoder_config_init(ma_format_f32, 1, 48000);
    ma_decoder dec;
    int res = ma_decoder_init_file("../candy.wav", &decconf, &dec);
    std::cout<<res <<std::endl;
    
    constexpr int readFrame = 1024*1000;
    constexpr float overlap = 0.0f;
    constexpr int windowRadix = 11;
    constexpr int windowSize = 1 << windowRadix;
    float *hostBuffer = new float[readFrame];
    ma_decoder_seek_to_pcm_frame(&dec, 48000*20);
    std::cout<<ma_decoder_read_pcm_frames(&dec, hostBuffer, readFrame, NULL) <<std::endl;
    
    
    
    constexpr int qt = toQuot(readFrame, overlap, windowSize);
    constexpr unsigned int OFullSize = qt * windowSize;
    constexpr unsigned int OHalfSize = OFullSize / 2;
    occa::memory dataIn = dev.malloc<float>(readFrame);
    occa::memory dataOut = dev.malloc<float>(OHalfSize);
    
    
    occa::memory FReal = dev.malloc<float>(OFullSize);
    occa::memory FImag = dev.malloc<float>(OFullSize);
    occa::memory SReal = dev.malloc<float>(OFullSize);
    occa::memory SImag = dev.malloc<float>(OFullSize);
    
    occa::memory Rout = dev.malloc<float>(OFullSize);
    occa::memory Iout = dev.malloc<float>(OFullSize);
    
    dataIn.copyFrom(hostBuffer);
    
    occa::kernel overlap_common = dev.buildKernel("../include/RadixCommon.okl", "Overlap_Common", prop);
    
    occa::kernel Butterfly_Common = dev.buildKernel("../include/RadixCommon.okl", "StockHamDITCommon", prop);
    occa::kernel bitReverseTemp = dev.buildKernel("../include/kernel.okl", "bitReverse_temp", prop);
    
    
    occa::kernel R6 = dev.buildKernel("../include/Radix6.okl", "Stockhpotimized6", prop);
    occa::kernel R7 = dev.buildKernel("../include/Radix7.okl", "Stockhpotimized7", prop);
    occa::kernel R8 = dev.buildKernel("../include/Radix8.okl", "Stockhpotimized8", prop);
    occa::kernel R9 = dev.buildKernel("../include/Radix9.okl", "Stockhpotimized9", prop);
    occa::kernel R10 = dev.buildKernel("../include/Radix10.okl", "Stockhpotimized10", prop);
    occa::kernel R11 = dev.buildKernel("../include/Radix11.okl", "Stockhpotimized11", prop);
    occa::kernel R10AIO=dev.buildKernel("../include/Radix10.okl", "preprocessed_ODW10_STH_STFT", prop);
    occa::kernel R11AIO=dev.buildKernel("../include/Radix11.okl", "preprocessed_ODW11_STH_STFT", prop);
    overlap_common( dataIn, 
                    OFullSize, 
                    readFrame, 
                    windowRadix, 
                    (unsigned int)(windowSize * (1.0f - overlap)), 
                    FReal
                    );
    overlap_common( dataIn, 
                    OFullSize, 
                    readFrame, 
                    windowRadix, 
                    (unsigned int)(windowSize * (1.0f - overlap)), 
                    Rout
                    );
                    
    R11
    (
        Rout,
        Iout,
        OHalfSize
    );
    // R11AIO
    // (
    //     dataIn,
    //     qt,
    //     readFrame,
    //     (unsigned int)(windowSize * (1.0f - overlap)),
    //     OHalfSize,
    //     Rout,
    //     Iout
    // );
    R10
    (
        FReal,
        FImag,
        OHalfSize
    );
    Butterfly_Common
    (
        FReal,
        FImag,
        SReal,
        SImag,
        (windowSize / 2),
        10,
        OHalfSize,
        windowRadix
    );
    // for(unsigned int stage = 0; stage < windowRadix; ++stage)
    // {
    //     if(stage % 2 == 0)
    //     {
    //         Butterfly_Common
    //         (
    //             FReal,
    //             FImag,
    //             SReal,
    //             SImag,
    //             (windowSize / 2),
    //             stage,
    //             OHalfSize,
    //             windowRadix
    //         );
    //         std::cout<<"touched"<<std::endl;
    //         getchar();
    //     }
    //     else
    //     {
    //         Butterfly_Common
    //         (
    //             SReal,
    //             SImag,
    //             FReal,
    //             FImag,
    //             (windowSize / 2),
    //             stage,
    //             OHalfSize,
    //             windowRadix
    //         );
    //     }

    // }
    std::vector<float> fixedOut(OFullSize);
    std::vector<float> commonOut(OFullSize);
    
    Rout.copyTo(fixedOut.data());
    if(windowRadix % 2 == 0)
    {
        FReal.copyTo(commonOut.data());
    }
    else
    {
        SReal.copyTo(commonOut.data());
    }
    for(unsigned int i =0; i< OFullSize; ++i)
        {
            if(fixedOut[i] != commonOut[i])
            {
                std::cout << "DIFF: " << fixedOut[i] - commonOut[i] <<std::endl;
            }
        }
    
    
    float * Rfixed = new float[OFullSize];
    float * Ifixed = new float[OFullSize];

    float * Rcommon = new float[OFullSize];
    float * Icommon = new float[OFullSize];
    
    auto compareHBuf = new occacplx[OFullSize];
    dev.finish();
    
    Rout.copyTo(Rfixed);
    Iout.copyTo(Ifixed);
    if(windowRadix % 2 == 0)
    {
        FReal.copyTo(Rcommon);
        FImag.copyTo(Icommon);
    }
    else
    {
        SReal.copyTo(Rcommon);
        SImag.copyTo(Icommon);
    }
    
    
    std::vector<float> mus_data;
    for(int i = 0;i<qt;++i)
    {
        ComplexVector cv(windowSize);
        for(int j=0;j<windowSize;++j)
        {
            cv[j].real(Rcommon[i*windowSize + j]);
            cv[j].imag(Icommon[i*windowSize + j]);
        }
        auto result = ifft(cv);
        
        for(auto j : result)
        {
        
            mus_data.push_back(j.real()/(windowSize));
        }
        
    }

    ma_device_config devconf = ma_device_config_init(ma_device_type_playback);
    ma_device mdevice;
    devconf.playback.channels = 1;
    devconf.sampleRate = 48000;
    devconf.playback.format=ma_format_f32;
    devconf.dataCallback = data_callback;
    
    devconf.pUserData = mus_data.data();
    ma_device_init(NULL, &devconf, &mdevice);
    ma_device_start(&mdevice);

    getchar();
}
