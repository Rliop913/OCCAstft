
#include "clStruct.hpp"

void
Runner::InitEnv()
{
    env = new Genv;
    env->PF = clboost::get_platform();
    env->DV = clboost::get_gpu_device(env->PF);
    env->CT = clboost::get_context(env->DV);
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
    Program codeBase = clboost::make_prog(clCodes.opencl_code, env->CT, env->DV);
    
    kens->EXP6STFT = clboost::make_kernel(codeBase, "_occa_Stockhoptimized6_0");
    kens->EXP7STFT = clboost::make_kernel(codeBase, "_occa_Stockhoptimized7_0");
    kens->EXP8STFT = clboost::make_kernel(codeBase, "_occa_Stockhoptimized8_0");
    kens->EXP9STFT = clboost::make_kernel(codeBase, "_occa_Stockhoptimized9_0");
    kens->EXP10STFT = clboost::make_kernel(codeBase, "_occa_Stockhoptimized10_0");
    kens->EXP11STFT = clboost::make_kernel(codeBase, "_occa_Stockhoptimized11_0");
    
    kens->EXPCommon = clboost::make_kernel(codeBase, "_occa_StockHamDITCommon_0");
    kens->Overlap = clboost::make_kernel(codeBase, "_occa_Overlap_Common_0");
    kens->DCRemove = clboost::make_kernel(codeBase, "_occa_DCRemove_Common_0");
    
    kens->Hanning = clboost::make_kernel(codeBase, "_occa_Window_Hanning_0");
    kens->Hamming = clboost::make_kernel(codeBase, "_occa_Window_Hamming_0");
    kens->Blackman = clboost::make_kernel(codeBase, "_occa_Window_Blackman_0");
    kens->Blackman_Harris = clboost::make_kernel(codeBase, "_occa_Window_Blackman_harris_0");
    kens->Blackman_Nuttall = clboost::make_kernel(codeBase, "_occa_Window_Blackman_Nuttall_0");
    kens->Nuttall = clboost::make_kernel(codeBase, "_occa_Window_Nuttall_0");
    kens->FlatTop = clboost::make_kernel(codeBase, "_occa_Window_FlatTop_0");
    kens->Gaussian = clboost::make_kernel(codeBase, "_occa_Window_Gaussian_0");

    kens->HalfComplex = clboost::make_kernel(codeBase, "_occa_toHalfComplexFormat_0");
    
    kens->toPower = clboost::make_kernel(codeBase, "_occa_toPower_0");

}

MAYBE_DATA
Runner::ActivateSTFT(   VECF& inData, 
                        const int& windowSizeEXP, 
                        const float& overlapRatio,
                        const std::string& options)
{
    const unsigned int FullSize = inData.size();
    const int windowSize = 1 << windowSizeEXP;
    const int qtConst = toQuot(FullSize, overlapRatio, windowSize);
    const unsigned int OFullSize = qtConst * windowSize;
    const unsigned int OHalfSize = OFullSize / 2;
    const unsigned int OMove     = windowSize * (1.0f - overlapRatio);

    CommandQueue CQ = clboost::make_cq(env->CT, env->DV);
    clData cldta;
    cldta.cq = &CQ;
    cldta.env= env;
    cldta.kens=kens;

    Buffer inMem = clboost::HTDCopy<float>(env->CT, FullSize, inData.data());
    Buffer RealMem = clboost::DMEM<float>(env->CT, OFullSize);
    Buffer ImagMem = clboost::DMEM<float>(env->CT, OFullSize);
    Buffer subReal;
    Buffer subImag;
    Buffer outMem = Buffer(env->CT, CL_MEM_WRITE_ONLY, sizeof(float)* OFullSize);
    
    Buffer* Rout = &RealMem;
    Buffer* Iout = &ImagMem;
    std::string res =
    runnerFunction::Default_Pipeline
    (
        &cldta,
        &inMem,
        &RealMem,
        &ImagMem,
        &subReal,
        &subImag,
        &outMem,
        std::move(FullSize),
        std::move(windowSize),
        std::move(qtConst),
        std::move(OFullSize),
        std::move(OHalfSize),
        std::move(OMove),
        options,
        windowSizeEXP,
        overlapRatio
    );

    if(res != "OK")
    {
        return std::nullopt;
    }
    std::vector<float> outData(OFullSize);
    if(clboost::q_read(CQ, outMem, true, OFullSize, outData.data()) != CL_SUCCESS)
    {
        return std::nullopt;
    }
    
    return std::move(outData);
}