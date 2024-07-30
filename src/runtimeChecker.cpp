#include "runtimeChecker.hpp"

bool
RuntimeCheck::isAvailable(const PATH& path)
{

    return true;
}


MAYBE_PATH
FallbackList::getNext()
{
    PATH result;
    if (VectorITR<CUDA>(CUDAFallback, result))
    {
        return result;
    }
    else if (VectorITR<OPENCL>(OpenCLFallback, result))
    {
        return result;
    }
    else if (VectorITR<OPENMP>(OpenMPFallback, result))
    {
        return result;
    }
    else if (VectorITR<SERVER>(ServerFallback, result))
    {
        return result;
    }
    else if (VectorITR<SERIAL>(SerialFallback, result))
    {
        return result;
    }
    return std::nullopt;
}

