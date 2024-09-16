#include "cl_global_custom.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<Platform>
clboost::get_platform()
{
	vector<Platform> pfvec;
	Platform::get(&pfvec);
	ASSERT_UEQ(pfvec.empty(), true);
	// for (int i = 0; i < pfvec.size(); i++) {

	// 	std::string pname;
	// 	vector<Device> dvs;
	// 	pfvec[i].getInfo(CL_PLATFORM_NAME, &pname);
	// 	pfvec[i].getDevices(CL_DEVICE_TYPE_ALL, &dvs);
	// 	for (int j = 0; j < dvs.size(); j++) {
	// 		std::string dname;
	// 		dvs[j].getInfo(CL_DEVICE_NAME, &dname);
	// 		std::cout<< dname<<std::endl;
	// 	}
	// }

	return pfvec;
}

Device
clboost::get_gpu_device(const std::vector<Platform>& pf)
{
	int dvScore = 0;
	Device outDevice;
	// std::string dname;
	// int maxGroupSize = 0;
	for(auto platform : pf){
		vector<Device> dvs;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &dvs);
		for(auto device : dvs){
			int Score=0;
			device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &Score);
			// device.getInfo(CL_DEVICE_NAME, &dname);
			// device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxGroupSize);
			if(Score > dvScore){
				outDevice = device;
				dvScore = Score;
			}
		}
	}
	return outDevice;
	// ASSERT_UEQ(dvs.empty(), true);
	// Device dv = dvs[0];
	//return dv;
}

Context
clboost::get_context(const Device& dev)
{
	return Context(dev);
}

CommandQueue
clboost::make_cq(const Context& ct, const Device& dev)
{
	return CommandQueue(ct, dev);
}
#ifndef NO_EMBEDDED_CL


Program
clboost::make_prog(const std::string& hardcoded_cl_code, const Context& ct, const Device& dev)
{
	Program pg(ct, hardcoded_cl_code);
	
	auto buildSuccess = pg.build(dev);
	std::string err_message = pg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
	std::cout << err_message << std::endl;
	ASSERT_EQ(buildSuccess, CL_SUCCESS);
	//ASSERT_EQ(err_message, "\n");
	return pg;
}
#endif  // !NO_EMBEDDED_CL
#ifdef NO_EMBEDDED_CL


Program
clboost::make_prog(const std::string& path, const Context& ct, const Device& dev)
{

	ASSERT_NO_STRING(path);
	std::ifstream fp;
	fp.open(path);
	ASSERT_EQ(fp.is_open(), true);
	std::stringstream ss;
	ss << fp.rdbuf();
	string source=ss.str();
	ASSERT_NO_STRING(source);
	
	Program pg(ct, source);

	pg.build(dev);

	ASSERT_EQ(pg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev), "\n");
	return pg;
}
#endif  // NO_EMBEDDED_CL
Kernel 
clboost::make_kernel(const Program& prog, const std::string& class_name)
{
	return Kernel(prog, class_name.c_str());
}



cl_int 
clboost::enq_q(CommandQueue& q, const Kernel& kernel, const int global_size, const int local_size)
{
	ASSERT_UEQ(global_size, 0);
	return q.enqueueNDRangeKernel(kernel, NullRange, NDRange(global_size), (local_size == -1 ? NullRange : NDRange(local_size)));
	
}


