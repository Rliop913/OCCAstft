#include "CL_Wrapper.h"

struct Program_Cuda
{
    void program_buffer_load();
    void program_execute();
    void program_uninit();
};

struct GPGPU
{
private:
    CL_env env;
public:
    CL_Program program;
    void build(const std::string& CL_Kcode, const std::string& entry);
    GPGPU(const std::string& platform, const std::string& device);
};


GPGPU::GPGPU(const std::string& platform, const std::string& device)
{
    std::vector<Platform> pvec;
    Platform::get(&pvec);
    for (int i = 0; i < pvec.size(); ++i) {
        std::string pname;
        pvec[i].getInfo(CL_PLATFORM_NAME, &pname);
        if (pname == platform) {
            env.PF = pvec[i];
            break;
        }
    }
    std::vector<Device> dvec;
    env.PF.getDevices(CL_DEVICE_TYPE_ALL, &dvec);
    for (int i = 0; i < dvec.size(); ++i) {
        std::string dname;
        dvec[i].getInfo(CL_DEVICE_NAME, &dname);
        if (dname == device) {
            env.DV = dvec[i];
            break;
        }
    }
    env.CT = Context(env.DV);
}


inline
void
GPGPU::build(const std::string& CL_Kcode, const std::string& entry)
{
    program.build(env, CL_Kcode, entry);
}