#pragma once
#include <string>
#include "CL/opencl.hpp"
#include "custom_assert.h"
#include <vector>
using namespace cl;
namespace clboost {

	Platform get_platform();
	Device get_gpu_device(const Platform& pf);
	Context get_context(const Device& dev);
	CommandQueue make_cq(const Context& ct,const Device& dev);
	Program make_prog(const std::string& path,const Context& ct,const Device& dev);

	template <typename T>
	Buffer HTDCopy(const Context& ct, const int& size, std::vector<T>& vec);
	template <typename T>
	Buffer HTDCopy(const Context& ct, const int& size, T* data);
	template <typename W>
	Buffer make_w_buf(const Context& ct, const int& size);
	template <typename TEMP>
	Buffer DMEM(const Context& ct, const int& size);
	Kernel make_kernel(const Program& prog,const std::string& class_name);

	template <typename... Args>
	void set_args(Kernel& kn, const Args ... args);

	cl_int enq_q(CommandQueue& q, const Kernel& kernel, const int global_size, const int local_size = NULL);

	template<typename P>
	cl_int q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, std::vector<P>& data);
	
	template<typename P>
	cl_int q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, P* data);

	//template<typename P>
	//void q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, vector<P>& data, Event& this_event, const vector<Event>& wait_ev);

}


template <typename T>
Buffer
clboost::HTDCopy(const Context& ct, const int& size, std::vector<T>& vec)
{
	return Buffer(ct, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * size, vec.data());
}

template <typename T>
Buffer
clboost::HTDCopy(const Context& ct, const int& size, T* data)
{
	return Buffer(ct, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * size, data);
}

template<typename W>
Buffer
clboost::make_w_buf(const Context& ct, const int& size)
{
	return Buffer(ct, CL_MEM_WRITE_ONLY, sizeof(W) * size);
}

template<typename TEMP>
Buffer
clboost::DMEM(const Context& ct, const int& size)
{
	return Buffer(ct, CL_MEM_READ_WRITE, sizeof(TEMP)* size);
}


template <typename... Args>
void
clboost::set_args(Kernel& kn, const Args ... args)
{
	int index = 0;
	(kn.setArg(index++, args),...);
	//(ASSERT_EQ(kn.setArg(index++, args),0), ...);
	ASSERT_UEQ(index, 0);
}

template <typename P>
cl_int
clboost::q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, std::vector<P>& data)
{
	return q.enqueueReadBuffer(wbuf, (check_dirct ? CL_TRUE : CL_FALSE), 0, sizeof(P) * size, data.data());
}
//
//template <typename P>
//void
//clboost::q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, vector<P>& data, Event& this_event, const vector<Event>& wait_ev)
//{
//	q.enqueueReadBuffer(wbuf, (check_dirct ? CL_TRUE : CL_FALSE), 0, sizeof(P) * size, data.data(), wait_ev, this_event);
//}

template<typename P>
cl_int
clboost::q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, P* data)
{	
	//ASSERT_EQ(q.enqueueReadBuffer(wbuf, (check_dirct ? CL_TRUE : CL_FALSE), 0, sizeof(P) * size, data),0);
	return q.enqueueReadBuffer(wbuf, (check_dirct ? CL_TRUE : CL_FALSE), 0, sizeof(P) * size, data);
	//getchar();
}
