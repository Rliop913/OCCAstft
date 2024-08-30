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
    // ma_decoder *ddec = (ma_decoder*)pDevice->pUserData;
    // ma_decoder_read_pcm_frames(ddec, pOutput, frameCount, NULL);
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
    
    
    occa::dtype_t complex;
    complex.registerType();
    complex.addField("real", occa::dtype::float_);
    complex.addField("imag", occa::dtype::float_);
    
    constexpr int qt = toQuot(readFrame, overlap, windowSize);
    constexpr unsigned int OFullSize = qt * windowSize;
    constexpr unsigned int OHalfSize = OFullSize / 2;
    occa::memory dev_in = dev.malloc<float>(readFrame);
    occa::memory dev_out = dev.malloc<float>(OHalfSize);
    occa::memory stkout = dev.malloc<float>(OHalfSize);
    
    occa::memory FReal = dev.malloc<float>(OFullSize);
    occa::memory FImag = dev.malloc<float>(OFullSize);
    occa::memory SReal = dev.malloc<float>(OFullSize);
    occa::memory SImag = dev.malloc<float>(OFullSize);
    
    occa::memory Rout = dev.malloc<float>(OFullSize);
    occa::memory Iout = dev.malloc<float>(OFullSize);

    occa::memory compareBuffer = dev.malloc(OFullSize, complex);
    occa::memory compareOut = dev.malloc(OFullSize, complex);

    occa::memory compareAwindowOut = dev.malloc<float>(OFullSize);    
    std::vector<float> ftempf(OFullSize);
    
    dev_in.copyFrom(hostBuffer);
    FImag.copyFrom(ftempf.data());
    Iout.copyFrom(ftempf.data());
    
    occa::kernel overlap_common = dev.buildKernel("../include/RadixCommon.okl", "Overlap_Common", prop);
    
    occa::kernel overlap_N_window = dev.buildKernel("../include/kernel.okl", "overlap_N_window", prop);
    
    occa::kernel Butterfly_Common = dev.buildKernel("../include/RadixCommon.okl", "StockHamDITCommon", prop);
    occa::kernel Butterfly = dev.buildKernel("../include/kernel.okl", "Butterfly", prop);
    occa::kernel bitReverseTemp = dev.buildKernel("../include/kernel.okl", "bitReverse_temp", prop);
    
    occa::kernel toPower = dev.buildKernel("../include/kernel.okl", "toPower", prop);
    
    occa::kernel AIO = dev.buildKernel("../include/Radix11.okl", "preprocessed_ODW11_STH_STFT", prop);
    occa::kernel APP = dev.buildKernel("../include/Radix10.okl", "preprocesses_ODW_10", prop);
    
    overlap_common( dev_in, 
                    OFullSize, 
                    readFrame, 
                    windowRadix, 
                    (unsigned int)(windowSize * (1.0f - overlap)), 
                    FReal
                    );
    APP
    (
        dev_in,
        qt,
        readFrame,
        (unsigned int)(windowSize * (1.0f - overlap)),
        compareAwindowOut
    );
    std::vector<float> Aout(OFullSize);
    std::vector<float> Bout(OFullSize);
    FReal.copyTo(Aout.data());
    compareAwindowOut.copyTo(Bout.data());
    for(unsigned int i =0; i< OFullSize; ++i)
        {
            if(Aout[i] != Bout[i])
            {
                std::cout << "overlapDiff: " << Aout[i] - Bout[i] <<std::endl;
            }
        }
    // for(int stageRadix = 0; stageRadix < windowRadix; stageRadix+=2)
    // {
    //     Butterfly_Common
    //     (   
    //         FReal,
    //         FImag,
    //         SReal,
    //         SImag,
    //         windowSize / 2,
    //         stageRadix,
    //         OHalfSize,
    //         windowRadix
    //     );
    //     Butterfly_Common
    //     (   
    //         SReal,
    //         SImag,
    //         FReal,
    //         FImag,
    //         windowSize / 2,
    //         (stageRadix + 1),
    //         OHalfSize,
    //         windowRadix
    //     );
    // }
    
    
    float * Ropti = new float[OFullSize];
    float * Iopti = new float[OFullSize];

    float * ARopti = new float[OFullSize];
    float * AIopti = new float[OFullSize];
    
    auto compareHBuf = new occacplx[OFullSize];

    AIO
    (
        dev_in,
        qt,
        readFrame,
        (unsigned int)(windowSize * (1.0f - overlap)),
        OHalfSize,
        Rout,
        Iout
    );

    // sthm11(stkbuf, OHalfSize);
    overlap_N_window
    (
        dev_in, 
        compareBuffer,
        readFrame,
        OFullSize,
        windowSize,
        (unsigned int)(windowSize * (1.0f - overlap))
    );
    // AIO(dev_in, qt, readFrame, (unsigned int)(windowSize * (1.0f - overlap)), OHalfSize, stkbuf);
    
    // Stockopt(stkbuf, OHalfSize);
    // StockHam(stkbuf, OHalfSize);
    // optimizedDIFBUTTERFLY(stkbuf, OHalfSize);
    // bitReverseTemp(dev_buffer, stkbufout, OFullSize, windowSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 1, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 2, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 4, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 8, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 16, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 32, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 64, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 128, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 256, OHalfSize, windowRadix);
    // Butterfly(stkbufout, windowSize, 512, OHalfSize, windowRadix);
    // Butterfly(dev_buffer, windowSize, 16384, OHalfSize, windowRadix);
    // Butterfly(dev_buffer, windowSize, 8192, OHalfSize, windowRadix);
    // Butterfly(dev_buffer, windowSize, 4096, OHalfSize, windowRadix);
    // Butterfly(dev_buffer, windowSize, 2048, OHalfSize, windowRadix);
    // Butterfly(dev_buffer, windowSize, 1024, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 512, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 256, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 128, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 64, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 32, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 16, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 8, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 4, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 2, OHalfSize, windowRadix);
    Butterfly(compareBuffer, windowSize, 1, OHalfSize, windowRadix);
    bitReverseTemp(compareBuffer, compareOut, OFullSize, windowSize, windowRadix);
    // bitReverseTemp(stkbuf, stkbufout, OFullSize, windowSize, windowRadix);
    dev.finish();
    // stkbufout.copyTo(opti);
    FReal.copyTo(Ropti);
    FImag.copyTo(Iopti);
    Rout.copyTo(ARopti);
    Iout.copyTo(AIopti);
    
    compareOut.copyTo(compareHBuf);
    
    for(unsigned int i =0; i< OFullSize; ++i)
    {
        if(ARopti[i] != compareHBuf[i].real || AIopti[i] != compareHBuf[i].imag)
        {
            std::cout <<i << " DIFF: " << compareHBuf[i].real << " AND " << ARopti[i] <<std::endl;
        }
    }
    
    // stkbufout.copyTo(Butterout);
    std::vector<float> mus_data;
    for(int i = 0;i<qt;++i)
    {
        ComplexVector cv(windowSize);
        for(int j=0;j<windowSize;++j)
        {
            cv[j].real(ARopti[i*windowSize + j]);
            cv[j].imag(AIopti[i*windowSize + j]);
        }
        auto result = ifft(cv);
        
        for(auto j : result)
        {
        
            mus_data.push_back(j.real()/(windowSize));
        }
        
    }
    for(int i = 0;i<OFullSize;++i)
    {
        // mus_data[i] = result[i].real();
        //std::cout<<"IDX: "<<i<<" DATA: "<<mus_data[i]<<std::endl;
    }

    ma_device_config devconf = ma_device_config_init(ma_device_type_playback);
    ma_device mdevice;
    devconf.playback.channels = 1;
    devconf.sampleRate = 48000;
    devconf.playback.format=ma_format_f32;
    devconf.dataCallback = data_callback;
    //devconf.pUserData = &dec;
    devconf.pUserData = mus_data.data();
    ma_device_init(NULL, &devconf, &mdevice);
    ma_device_start(&mdevice);

    getchar();


    // return 0;
    // toPower(dev_buffer, dev_out, OHalfSize, windowRadix);
    // ken(    dev_in, 
    //         dev_out,
    //         dev_buffer,
    //         readFrame,
    //         1024,
    // //         qt,
    // //         512,
    // //         10);
    // delete[] hostBuffer;
    // float* output = new float[OHalfSize];
    // dev_out.copyTo(output);
    
    // for(int i=0;i<10;++i)//csv out
    // {
    //     for(int j=0;j<windowSize/2;++j)
    //     {
    //         std::cout<<output[i*windowSize/2+j]<<",";
    //     }
    //     std::cout<<"0"<<std::endl;
    // }
    // delete[] output;
}
