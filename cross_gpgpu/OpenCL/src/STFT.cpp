#include "STFT.hpp"
#include "CL_Wrapper.h"
#include "cl_global_custom.h"



void
STFT::InitEnv()
{
    env.PF = clboost::get_platform();
    env.DV = clboost::get_gpu_device(env.PF);
    env.CT = clboost::get_context(env.DV);
    env.CQ = clboost::make_cq(env.CT);
}

STFT::STFT()
{
    Init();
}



int
main()
{
    
    return 0;
}