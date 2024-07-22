#include "STFT.h"
#include <cmath>
std::vector<std::pair<std::string, std::string>>
OPENCL_ACC::get_platform_device_list()
{
    std::vector<std::pair<std::string, std::string>> pla_dev;
    std::vector<Platform> pvec;
    Platform::get(&pvec);
    for (int i = 0; i < pvec.size(); ++i) {
        std::string pname;
        std::vector<Device> dvec;
        pvec[i].getInfo(CL_PLATFORM_NAME, &pname);
        pvec[i].getDevices(CL_DEVICE_TYPE_ALL, &dvec);
        for (int j = 0; j < dvec.size(); ++j) {
            std::pair<std::string, std::string> dpair;
            std::string dname;
            dvec[j].getInfo(CL_DEVICE_NAME, &dname);
            dpair.first = pname;
            dpair.second = dname;
            pla_dev.push_back(dpair);
        }
    }
    return pla_dev;
}
void
OPENCL_ACC::set_platform_device(const std::string& platform, const std::string& device)
{
    std::vector<Platform> pvec;
    Platform::get(&pvec);
    for (int i = 0; i < pvec.size(); ++i) {
        std::string pname;
        pvec[i].getInfo(CL_PLATFORM_NAME, &pname);
        
        if (pname == platform) {
            this->PF = pvec[i];
            break;
        }
    }
    std::vector<Device> dvec;
    this->PF.getDevices(CL_DEVICE_TYPE_ALL, &dvec);
    for (int i = 0; i < dvec.size(); ++i) {
        std::string dname;
        dvec[i].getInfo(CL_DEVICE_NAME, &dname);
        if (dname == device) {
            this->DV = dvec[i];
            
            break;
        }
    }
    this->CT = Context(this->DV);
    this->ready = true;
    KS = cl_facade::create_kernel(CLS->STFT, "STFT", CT, DV);
    CO = clboost::make_cq(CT, DV);
}


OPENCL_ACC::OPENCL_ACC(const std::string& platform, const std::string& device)
{
    CLS = new cl_embed();
    set_platform_device(platform, device);
}
OPENCL_ACC::OPENCL_ACC() {
    CLS = new cl_embed();
}
OPENCL_ACC::~OPENCL_ACC() {
    delete CLS;
}


std::vector<float>
OPENCL_ACC::cl_STFT(short* full_frame, const ma_uint64& origin_len, const int& window_radix_2, const double& overlap_ratio, const int& front_side_z_padding_size, int& number_of_frames)
{

    
    int window_len = std::pow(2, (int)window_radix_2);
    int overlaped_frame = (1.0 - overlap_ratio) * window_len;
    int fft_quotient = origin_len / (int)((double)window_len * (1.0 - overlap_ratio));
    number_of_frames = fft_quotient;
    cl_uint overlaped_full_frame = fft_quotient * window_len;
    cl_float2* temp_buffer = nullptr;

	Buffer in = clboost::make_r_buf<short>(CT, origin_len, full_frame);
    Buffer innerTemp = clboost::make_temp_buf<cl_float2>(CT, overlaped_full_frame);
	Buffer out = clboost::make_w_buf<float>(CT, overlaped_full_frame / 2);
	
    error_arr era(9);
    int errCount = 0;
    era[errCount++] = KS.setArg(0, in);
    era[errCount++] = KS.setArg(1, innerTemp);
    era[errCount++] = KS.setArg(2, out);
    era[errCount++] = KS.setArg(3, overlaped_frame);
    era[errCount++] = KS.setArg(4, (unsigned int)origin_len);
    era[errCount++] = KS.setArg(5, front_side_z_padding_size);
    era[errCount++] = KS.setArg(6, window_radix_2);
    era[errCount++] = clboost::enq_q(CO, KS, overlaped_full_frame, window_len);
    vector<float> outSTFT(overlaped_full_frame / 2);
    era[errCount++] = clboost::q_read<float>(CO, out, true, overlaped_full_frame/2, outSTFT);
    return outSTFT;
	
}




// void
// OPENCL_ACC::to_dbfs(cl_float3* data, const int& window_radix_size, const int& low, const int& mid, const int& high, const int& quot )
// {
//     // int wos = pow(2, window_radix_size);
//     // gpgpu_facade<cl_float3, cl_float3>(
//     //     CLS->to_dbfs,
//     //     "to_dbfs",
//     //     data,
//     //     quot,
//     //     data,
//     //     quot,
//     //     quot,
//     //     wos,
//     //     low,
//     //     mid,
//     //     high
//     // );
    
// }











