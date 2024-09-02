#include "runtimeChecker.hpp"
#include <iostream>
#ifdef OS_WINDOWS
#include <process.h>
bool
RuntimeCheck::ExcuteRunner(const std::string& executePath, const int& portNum)
{
    std::string ptnum = std::to_string(portNum);
    auto exePath = fs::relative(executePath).string();
    char *argv[] = {(char*)exePath.c_str(), (char*)ptnum.c_str(), NULL};

    auto result = _spawnv(_P_NOWAIT, exePath.c_str(), argv);
    if(result == -1)
    {
        return false;
    }
    return true;
}



#else
extern char **environ;

bool
RuntimeCheck::ExcuteRunner(const std::string& executePath, const int& portNum)
{
    pid_t pid;
    std::string ptnum = std::to_string(portNum);
    auto exePath = fs::relative(executePath).string();

    char *argv[] = {(char*)exePath.c_str(), (char*)ptnum.c_str(), NULL};
    if (posix_spawn(&pid, exePath.c_str(), NULL, NULL, argv, environ) != 0)
    {
        return false;
    }
    return true;
}


#endif





bool
RuntimeCheck::isAvailable(PATH& path)
{
#ifdef OS_WINDOWS
    std::string dllName;
    fs::path executePath(path.second);
    if(executePath.extension() == "exe")
    {
        executePath.replace_extension("");
    }
    switch (path.first)
    {
        case SupportedRuntimes::CUDA:
            dllName = "nvcuda.dll";
            if(executePath.filename() != "cudaRun")
            {
                executePath.append("cudaRun.exe");
            }
            break;
        case SupportedRuntimes::HIP:
            dllName = "amdhip64.dll";
            if(executePath.filename() != "hipRun")
            {
                executePath.append("hipRun.exe");
            }
            break;
        case SupportedRuntimes::METAL:
            dllName = "SKIP";
            if(executePath.filename() != "metalRun")
            {
                executePath.append("metalRun.exe");
            }
            break;
        case SupportedRuntimes::OPENCL:
            dllName = "OpenCL.dll";
            if(executePath.filename() != "openclRun")
            {
                executePath.append("openclRun.exe");
            }
            break;

        case SupportedRuntimes::OPENMP:
            dllName = "SKIP";
            if(executePath.filename() != "openmpRun")
            {
                executePath.append("openmpRun.exe");
            }
            break;

        case SupportedRuntimes::SERVER:
            dllName = "SKIP";
            break;

        case SupportedRuntimes::SERIAL:
            dllName = "SKIP";
            if(executePath.filename() != "serialRun")
            {
                executePath.append("serialRun.exe");
            }
            break;
        case SupportedRuntimes::CUSTOM:
            executePath.replace_extension(".exe");
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
    if(executePath.extension() == "exe")
    {
        executePath.replace_extension("");
    }
    switch (path.first)
    {
        case SupportedRuntimes::CUDA:
            soName = "libcuda.so";
            if(executePath.filename() != "cudaRun")
            {
                executePath.append("cudaRun");
            }
            break;
        case SupportedRuntimes::HIP:
            soName = "libamdhip64.so";
            if(executePath.filename() != "hipRun")
            {
                executePath.append("hipRun");
            }
            break;
        case SupportedRuntimes::METAL:
            soName = "SKIP";
            if(executePath.filename() != "metalRun")
            {
                executePath.append("metalRun");
            }
        case SupportedRuntimes::OPENCL:
            soName = "libOpenCL.so";
            if(executePath.filename() != "openclRun")
            {
                executePath.append("openclRun");
            }
            break;
        case SupportedRuntimes::OPENMP:
            soName = "SKIP";
            if(executePath.filename() != "openmpRun")
            {
                executePath.append("openmpRun");
            }
            break;

        case SupportedRuntimes::SERVER:
            soName = "SKIP";
            break;

        case SupportedRuntimes::SERIAL:
            soName = "SKIP";
            if(executePath.filename() != "serialRun")
            {
                executePath.append("serialRun");
            }
            break;
        case SupportedRuntimes::CUSTOM:
            soName = "SKIP";
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
            executePath =  fs::absolute(executePath);
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
    else if (VectorITR<HIP>(HIPFallback, result))
    {
        return result;
    }
    else if (VectorITR<METAL>(METALFallback, result))
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
    else if (VectorITR<CUSTOM>(CustomFallback, result))
    {
        return result;
    }
    return std::nullopt;
}

