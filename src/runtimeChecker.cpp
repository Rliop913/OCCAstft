#include "runtimeChecker.hpp"

#ifdef OS_WINDOWS


MAYBE_RUNNER&
RuntimeCheck::ExcuteRunner(const std::string& executePath)
{
    STARTUPINFO si;
    RUNNER_INFO ri;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&ri, sizeof(ri));

    bool result = CreateProcess
    (
        NULL, NULL, NULL, NULL,
        FALSE,
        0,
        NULL, NULL,
        &si,
        &ri
    );

    if(result)
    {
        CloseHandle(ri.hProcess);
        CloseHandle(ri.hThread);
        return std::nullopt;
    }


    return std::move(ri);
}



#else




#endif





bool
RuntimeCheck::isAvailable(PATH& path)
{
#ifdef OS_WINDOWS
    std::string dllName;
    fs::path executePath(path.second);
    switch (path.first)
    {
        case SupportedRuntimes::CUDA:
            dllName = "nvcuda.dll";
            executePath.append("cudaRun.exe");
            break;

        case SupportedRuntimes::OPENCL:
            dllName = "OpenCL.dll";
            executePath.append("openclRun.exe");
            break;

        case SupportedRuntimes::OPENMP:
            dllName = "SKIP";
            executePath.append("openmpRun.exe");
            break;

        case SupportedRuntimes::SERVER:
            dllName = "SKIP";
            break;

        case SupportedRuntimes::SERIAL:
            dllName = "SKIP";
            executePath.append("serialRun.exe");
            break;
        default:
            return false;
    }
    if(dllName != "SKIP")
    {
        HMODULE runtime = LoadLibrary(dllName.c_str());
        if(!runtime) 
        {
            return false;
        }
        FreeLibrary(runtime);
    }

    if(path.first != SupportedRuntimes::SERVER)
    {
        if(!fs::exists(executePath))
        {
            return false;
        }
        else
        {
            executePath =  fs::relative(executePath);
            path.second = executePath.string(); //change into executable path
        }
    }
    return true;
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
            executePath =  fs::relative(executePath);
            path.second = executePath.string(); //change into executable path
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

