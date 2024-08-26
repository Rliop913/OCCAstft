
#include <CL/opencl.hpp>
#include "RunnerInterface.hpp"
#include "CL_Wrapper.h"
#include "cl_global_custom.h"
#include "okl_embedded.h"
struct Genv{
    std::vector<Platform> PF;
    Device DV;
    Context CT;
    CommandQueue CQ;
};


struct Gcodes{
    Kernel R10STFT;
    Kernel R11STFT;
    Kernel toPower;
};

void
Runner::InitEnv()
{
    env = new Genv;
    env->PF = clboost::get_platform();
    env->DV = clboost::get_gpu_device(env->PF);
    env->CT = clboost::get_context(env->DV);
    env->CQ = clboost::make_cq(env->CT, env->DV);
}

void
Runner::UnInit()
{
    //nothing
}

void
Runner::BuildKernel()
{
    kens = new Gcodes;
    okl_embed clCodes;
    Program codeBase = clboost::make_prog(clCodes.compiled, env->CT, env->DV);
    kens->R10STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW10_STH_STFT_0");
    kens->R11STFT = clboost::make_kernel(codeBase, "_occa_preprocessed_ODW11_STH_STFT_0");
    kens->toPower = clboost::make_kernel(codeBase, "_occa_toPower_0");
    

}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowRadix, 
                        const float& overlapRatio)
{
    const unsigned int FullSize = inData.size();
    const int windowSize = 1 << windowRadix;
    const int qtConst = toQuot(FullSize, overlapRatio, windowSize);
    const unsigned int OFullSize = qtConst * windowSize;
    const unsigned int OHalfSize = OFullSize / 2;
    const unsigned int OMove     = windowSize * (1.0f - overlapRatio);
    Buffer inMem = clboost::HTDCopy<float>(env->CT, FullSize, inData.data());
    Buffer outMem = clboost::DMEM<float>(env->CT, OHalfSize);
    Buffer tempMem = clboost::DMEM<cplx_t>(env->CT, OFullSize);
    
    std::vector<int> error_container(30 + windowRadix);
    int error_itr = 0;
    Kernel* RAIO;
    int workGroupSize = 0;
    switch (windowRadix)
    {
    case 11:
        RAIO = &(kens->R11STFT);
        workGroupSize = 1024;
        break;
    case 10:
        RAIO = &(kens->R10STFT);
        workGroupSize = 512;
        break;
    default:
        break;
    }
    
    error_container[error_itr++] = RAIO->setArg(0, inMem);
    error_container[error_itr++] = RAIO->setArg(1, qtConst);
    error_container[error_itr++] = RAIO->setArg(2, FullSize);
    error_container[error_itr++] = RAIO->setArg(3, OMove);
    error_container[error_itr++] = RAIO->setArg(4, OHalfSize);
    error_container[error_itr++] = RAIO->setArg(5, tempMem);
    
    error_container[error_itr++] = kens->toPower.setArg(0, tempMem);
    error_container[error_itr++] = kens->toPower.setArg(1, outMem);
    error_container[error_itr++] = kens->toPower.setArg(2, OHalfSize);
    error_container[error_itr++] = kens->toPower.setArg(3, windowRadix);
    
    error_container[error_itr++] = clboost::enq_q(env->CQ, (*RAIO), OFullSize, workGroupSize);
    
    error_container[error_itr++] = clboost::enq_q(env->CQ, kens->toPower, OHalfSize, LOCAL_SIZE);
    std::vector<float> outData(OHalfSize);
    error_container[error_itr++] = clboost::q_read(env->CQ, outMem, true, OHalfSize, outData.data());

    for(auto Eitr : error_container){
        if(Eitr != CL_SUCCESS){
            return std::nullopt;
        }
    }

    return std::move(outData);
}






// #include <iostream>
// int
// main()
// {
//     STFT temp = STFT();
//     ma_decoder_config decconf = ma_decoder_config_init(ma_format_f32, 1, 48000);
//     ma_decoder dec;

//     int res = ma_decoder_init_file("../../../candy.wav", &decconf, &dec);
//     constexpr int readFrame = 1024*1000;
//     std::vector<float> hostBuffer(readFrame);
//     ma_decoder_seek_to_pcm_frame(&dec, 48000*20);
//     ma_decoder_read_pcm_frames(&dec, hostBuffer.data(), readFrame, NULL);
//     int windowRadix = 10;
//     float overlapRatio = 0.5f;
//     auto out = temp.ActivateSTFT(hostBuffer, windowRadix, overlapRatio);
//     if(out.has_value()){
//         const int windowSize = 1 << windowRadix;
//         auto tout = out.value();
//         for(int i=0;i<10;++i)//csv out
//         {
//             for(int j=0;j<windowSize/2;++j)
//             {
//                 float data=tout.at(i*windowSize/2+j); 
//                 std::cout<<data<<",";
//             }
//             std::cout<<std::endl;
//         }
//     }
//     return 0;
// }