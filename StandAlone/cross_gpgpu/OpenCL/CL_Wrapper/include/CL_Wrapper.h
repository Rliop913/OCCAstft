#pragma once
#define NO_EMBEDDED_CL
#include "cl_FACADE.h"
#include "cl_global_custom.h"

struct CL_env
{
    Platform PF;
    Device DV;
    Context CT;
    CommandQueue CQ;
};

struct CL_Program
{
    Kernel KN;
    vector<Buffer> dat_in;
    vector<Buffer> dat_out;

    template <typename... Args>
    int set_args(const Args &...args);
    inline void build(CL_env &env, const std::string &CL_Kcode, const std::string &entry);
    void execute();
};

void CL_Program::build(CL_env &env, const std::string &CL_Kcode, const std::string &entry)
{
    KN = cl_facade::create_kernel(CL_Kcode, entry, env.CT, env.DV);
}

template <typename... Args>
int CL_Program::set_args(const Args &...args)
{
    int arg_index = 0;
    for (auto i : dat_in)
    {
        KN.setArg(arg_index++, i);
    }
    (KN.setArg(arg_index++, args), ...); // Fold
    for (auto i : dat_out)
    {
        KN.setArg(arg_index++, i);
    }
}