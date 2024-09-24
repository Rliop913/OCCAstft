
#include "cl_FACADE.h"
using namespace clboost;

#ifndef NO_EMBEDDED_CL


Kernel
cl_facade::create_kernel(const std::string& kernelCode, const std::string& kernelName, const Context& ct, const Device& dev)
{
	return make_kernel(make_prog(kernelCode, ct, dev), kernelName);
}
#endif  // NO_EMBEDDED_CL
#ifdef NO_EMBEDDED_CL


Kernel
cl_facade::create_kernel(const std::string& path, const std::string& class_name, const Context& ct, const Device& dev)
{
	return make_kernel(make_prog(path, ct, dev,to_text), class_name);
}
#endif  // NO_EMBEDDED_CL