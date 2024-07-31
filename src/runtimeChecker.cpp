#include "runtimeChecker.hpp"

bool
RuntimeCheck::isAvailable(const PATH& path)
{
#ifdef OS_WINDOWS
    HMODULE runtime = LoadLibrary("nvcuda.dll");
    if(!runtime)
    {
        //no cuda
    }
    else
    {
        FreeLibrary(runtime);
    }
#else
    std::string soName;
    fs::path executePath(path.second);
    switch (path.first)
    {
        case SupportedRuntimes::CUDA:
            soName = "libcuda.so";
            executePath.append("cudaRun");
            break;

        case SupportedRuntimes::OPENCL:
            soName = "libOpenCL.so";
            executePath.append("openclRun");
            break;

        case SupportedRuntimes::OPENMP:
            soName = "SKIP";
            executePath.append("openmpRun");
            break;

        case SupportedRuntimes::SERVER:
            soName = "SKIP";
            break;

        case SupportedRuntimes::SERIAL:
            soName = "SKIP";
            executePath.append("serialRun");
            break;

        default:
            return false;
    }

    if(soName != "SKIP")
    {
        void *runtime = dlopen(soName.c_str(), RTLD_NOW);
        if(!runtime) 
        {
            return false;
        }
        dlclose(runtime);//exists
    }

    if(path.first != SupportedRuntimes::SERVER)
    {
        if(!fs::exists(executePath))
        {
            return false;
        }
        else
        {
            std::string command = "\"" + executePath.string() + "\" local";
            int result = system(command.c_str());//excute runner
            if(result != 0)
            {
                return false;
            }
        }
    }
    return true;
#endif
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

