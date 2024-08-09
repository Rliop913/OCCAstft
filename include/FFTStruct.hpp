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
using MAYBE_DATA    = std::optional<std::vector<float>>;
using MAYBE_MEMORY  = std::optional<std::string>;
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
    int windowRadix                 = 10;
    float overlapRate               = 0.0f;
    ULL dataLength   = -1;

    MAYBE_DATA data                 = std::nullopt;
    MAYBE_MEMORY sharedMemoryInfo   = std::nullopt;

    std::string __mappedID;
    void* __memPtr = nullptr;
    
    int __POSIX_FileDes = -1;
    void* __WINDOWS_HANDLEPtr = nullptr;

    ULL adjustToPage();
public:
    FFTRequest(){}
    FFTRequest(const int& WR, const float& OLR, ULL& counter)
    : windowRadix(WR), overlapRate(OLR)
    {
        __mappedID = std::to_string(counter++);
    };
    BIN Serialize();//will add capnproto
    void Deserialize(const BIN& binData);
    void MakeSharedMemory(const SupportedRuntimes& Type, CULL& dataSize);
    void SetData(std::vector<float>& data);
    MAYBE_DATA getData();
    std::string getID(){return __mappedID;}
};
