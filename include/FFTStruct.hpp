#pragma once
#include <vector>
#include <optional>
#include <string>
#include <sstream>
#ifdef OS_POSIX
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>


#endif
#ifdef OS_WINDOWS
#include <Windows.h>
#endif
#include "runtimeChecker.hpp"

using BIN           = std::string;
using SHMOBJ        = std::pair<void*, int>;
using MAYBE_DATA    = std::optional<std::vector<float>>;
using MAYBE_MEMORY  = std::optional<std::string>;
using MAYBE_SHOBJ   = std::optional<SHMOBJ>;
using CULL          = const unsigned long long;
using ULL           = unsigned long long;
static const 
std::string frontTags[3] = 
    { 
        "<<WINDOW____RADIX>>", 
        "<<OVERLAPED__SIZE>>",
        "<<DATA_____LENGTH>>"
    };
static const 
std::string backTags[6] = 
    {
        "<<WINHANDLE_FIELD>>",
        "<<POSIX_FD__FIELD>>",
        "<<MEMPOINTERFIELD>>",
        "<<ID________FIELD>>",
        "<<MEMORY____FIELD>>", 
        "<<DATA______FIELD>>"
    };//backward

#define TAG_SIZE 19



struct FFTRequest{
private:
    MAYBE_MEMORY sharedMemoryInfo   = std::nullopt;
    MAYBE_DATA data                 = std::nullopt;
    std::string __mappedID;
    void* __memPtr = nullptr;
    
    int __POSIX_FileDes = -1;
    void* __WINDOWS_HANDLEPtr = nullptr;
    template<typename T>
    ULL adjustToPage(const ULL& length);
public:
    int windowRadix                 = 10;
    float overlapRate               = 0.0f;
    ULL dataLength   = -1;
    
    FFTRequest(){}
    FFTRequest(const int& WR, const float& OLR, ULL& counter)
    : windowRadix(WR), overlapRate(OLR)
    {
        __mappedID = std::to_string(counter++);
    };
    BIN Serialize();//will add capnproto
    void Deserialize(const BIN& binData);
    void MakeSharedMemory(const SupportedRuntimes& Type, CULL& dataSize);
    MAYBE_MEMORY GetSharedMemPath(){return sharedMemoryInfo;}

    //Integrity check from received object
    void BreakIntegrity();
    bool CheckIntegrity();
    void SetData(std::vector<float>& data);
    [[nodiscard]]
    MAYBE_DATA FreeAndGetData();
    [[nodiscard]]
    MAYBE_DATA GetData();

    MAYBE_SHOBJ GetSHMPtr();
    void FreeSHMPtr(SHMOBJ& shobj);
    //completely unlink
    void FreeData();
    std::string getID(){return __mappedID;}
};


template<typename T>
ULL
FFTRequest::adjustToPage(const ULL& length)
{   
    ULL dataSize = length * sizeof(T);
    unsigned long long PageSize = sysconf(_SC_PAGESIZE);
    return ((dataSize + PageSize - 1) / PageSize) * PageSize;
}