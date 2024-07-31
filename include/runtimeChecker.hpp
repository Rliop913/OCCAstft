#pragma once
#include <string>
#include <optional>
#include <filesystem>
#include <vector>
#ifdef OS_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>

#endif

namespace fs = std::filesystem; 

enum SupportedRuntimes {
    CUDA,
    OPENCL,
    OPENMP,
    SERVER,
    SERIAL
};

using STRVEC        = std::vector<std::string>;
using PATH      = std::pair<SupportedRuntimes, std::string>;
using MAYBE_PATH= std::optional<PATH>;


namespace RuntimeCheck{
    bool isAvailable(const PATH& path);
    


};

//Calculation fallback lists.
struct FallbackList{
public:
    //checked first, reads from back. put excutable cudaRun's directory path here
    STRVEC CUDAFallback = {"./"};

    //checked second, reads from back. put excutable openclRun's directory path here
    STRVEC OpenCLFallback = {"./"};

    //checked third, reads from back. put excutable openmpRun's directory path here
    STRVEC OpenMPFallback = {"./"};

    //checked last, reads from back. put excutable serialRun's directory path here
    STRVEC SerialFallback = {"./"};

    //checked fourth, reads from back. put server IP or URL here, server should manually run STFT runtimes. 
    STRVEC ServerFallback;

    MAYBE_PATH  getNext();
private:
    template <SupportedRuntimes Type>
    bool VectorITR(STRVEC& vec, PATH& result);
};

template<SupportedRuntimes Type>
inline
bool
FallbackList::VectorITR(STRVEC& vec, PATH& result)
{
    for(;;)
    {
        if(vec.empty())
        {
            return false;
        }
        else
        {
            auto path = vec.back();
            vec.pop_back();
            if(fs::exists(path))
            {
                result.first = Type;
                result.second = path;
                return true;
            }
        }
    }
}


template<>
inline
bool
FallbackList::VectorITR<SupportedRuntimes::SERVER>(STRVEC& vec, PATH& result)
{
    for(;;)
    {
        if(vec.empty())
        {
            return false;
        }
        else
        {
            auto path = vec.back();
            vec.pop_back();
            result.first = SERVER;
            result.second = path;
            return true;
        }
    }
}



// struct ExePath{
//     std::string CUDAexe = "./cudaRun";
//     std::string OpenCLExe = "./openclRun";
//     std::string OpenMPExe = "./openmpRun";
//     std::string SerialExe = "./serialRun";
// };
