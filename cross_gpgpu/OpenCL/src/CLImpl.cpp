


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
    Kernel overlapNWindow;
    Kernel rmDC;
    Kernel bitReverse;
    Kernel endPreProcess;
    Kernel butterfly;
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
Runner::BuildKernel()
{
    kens = new Gcodes;
    okl_embed clCodes;
    Program codeBase = clboost::make_prog(clCodes.compiled_code, env->CT, env->DV);
    
    kens->rmDC = clboost::make_kernel(codeBase, "_occa_removeDC_0"); 
    // cl_facade::create_kernel(clCodes.compiled_code, "_occa_removeDC_0", env->CT, env->DV);

    kens->overlapNWindow = clboost::make_kernel(codeBase, "_occa_overlap_N_window_0");
    // cl_facade::create_kernel(clCodes.compiled_code, "_occa_overlap_N_window_0", env->CT, env->DV);

    kens->bitReverse = clboost::make_kernel(codeBase, "_occa_bitReverse_0");
    // cl_facade::create_kernel(clCodes.compiled_code, "_occa_bitReverse_0", env->CT, env->DV);

    kens->endPreProcess = clboost::make_kernel(codeBase, "_occa_endPreProcess_0");
    // cl_facade::create_kernel(clCodes.compiled_code, "_occa_endPreProcess_0", env->CT, env->DV);

    kens->butterfly = clboost::make_kernel(codeBase, "_occa_Butterfly_0");
    // cl_facade::create_kernel(clCodes.compiled_code, "_occa_Butterfly_0", env->CT, env->DV);

    kens->toPower = clboost::make_kernel(codeBase, "_occa_toPower_0");
    // cl_facade::create_kernel(clCodes.compiled_code, "_occa_toPower_0", env->CT, env->DV);

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
    Buffer qtBuffer = clboost::DMEM<float>(env->CT, qtConst);
    Buffer tempMem = clboost::DMEM<cplx_t>(env->CT, OFullSize);
    
    std::vector<int> error_container(30 + windowRadix);
    int error_itr = 0;

    error_container[error_itr++] = kens->overlapNWindow.setArg(0, inMem);
    error_container[error_itr++] = kens->overlapNWindow.setArg(1, tempMem);
    error_container[error_itr++] = kens->overlapNWindow.setArg(2, FullSize);
    error_container[error_itr++] = kens->overlapNWindow.setArg(3, OFullSize);
    error_container[error_itr++] = kens->overlapNWindow.setArg(4, windowSize);
    error_container[error_itr++] = kens->overlapNWindow.setArg(5, OMove);
    
    error_container[error_itr++] = kens->rmDC.setArg(0, tempMem);
    error_container[error_itr++] = kens->rmDC.setArg(1, OFullSize);
    error_container[error_itr++] = kens->rmDC.setArg(2, qtBuffer);
    error_container[error_itr++] = kens->rmDC.setArg(3, windowSize);
    
    error_container[error_itr++] = kens->bitReverse.setArg(0, tempMem);
    error_container[error_itr++] = kens->bitReverse.setArg(1, OFullSize);
    error_container[error_itr++] = kens->bitReverse.setArg(2, windowSize);
    error_container[error_itr++] = kens->bitReverse.setArg(3, windowRadix);
    
    error_container[error_itr++] = kens->endPreProcess.setArg(0, tempMem);
    error_container[error_itr++] = kens->endPreProcess.setArg(1, OFullSize);
    
    error_container[error_itr++] = kens->butterfly.setArg(0, tempMem);
    error_container[error_itr++] = kens->butterfly.setArg(1, windowSize);
    //idx 2 changes in stage loop
    error_container[error_itr++] = kens->butterfly.setArg(3, OHalfSize);
    error_container[error_itr++] = kens->butterfly.setArg(4, windowRadix);
    
    error_container[error_itr++] = kens->toPower.setArg(0, tempMem);
    error_container[error_itr++] = kens->toPower.setArg(1, outMem);
    error_container[error_itr++] = kens->toPower.setArg(2, OHalfSize);
    error_container[error_itr++] = kens->toPower.setArg(3, windowRadix);
    
    error_container[error_itr++] = clboost::enq_q(env->CQ, kens->overlapNWindow, OFullSize, LOCAL_SIZE);
    error_container[error_itr++] = clboost::enq_q(env->CQ, kens->rmDC, OFullSize, LOCAL_SIZE);
    error_container[error_itr++] = clboost::enq_q(env->CQ, kens->bitReverse, OFullSize, LOCAL_SIZE);
    error_container[error_itr++] = clboost::enq_q(env->CQ, kens->endPreProcess, OFullSize, LOCAL_SIZE);
    
    for(int iStage=0; iStage < windowRadix; ++iStage)
    {
        kens->butterfly.setArg(2, 1 << iStage);
        error_container[error_itr++] = clboost::enq_q(env->CQ, kens->butterfly, OHalfSize, LOCAL_SIZE);
    }
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