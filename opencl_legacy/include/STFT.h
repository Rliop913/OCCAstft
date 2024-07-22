#pragma once
#include "cl_FACADE.h"
#include "cl_global_custom.h"
#include "cl_embedded.h"
#include <vector>
#include <memory>
#define DEFAULT_SAMPLERATE 48000
#define to_big_radix_2(var) pow(2,std::ceil(log2(var)));
using error_arr = std::vector<cl_int>;
using TIDX = unsigned int;
typedef unsigned long long ma_uint64;


class OPENCL_ACC{
private:
	cl_embed *CLS;
	Kernel KS;
	CommandQueue CO;
	bool ready = false;
public:
	Platform PF;
	Device DV;
	Context CT;
	OPENCL_ACC();
	OPENCL_ACC(const std::string& platform, const std::string& device);
	~OPENCL_ACC();
	std::vector<std::pair<std::string, std::string>> get_platform_device_list();
	void set_platform_device(const std::string& platform, const std::string& device);

	std::vector<float> cl_STFT(short* full_frame, const ma_uint64& full_length, const int& window_radix_2, const double& overlap_ratio, const int& both_side_z_padding_size, int& number_of_index);
};

