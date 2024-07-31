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

#include "runtimeChecker.hpp"

using BIN           = std::string;
using MAYBE_DATA    = std::optional<std::vector<float>>;
using MAYBE_MEMORY  = std::optional<std::string>;

const std::string frontTags[3] = 
    { 
        "<<WINDOW____RADIX>>", 
        "<<OVERLAPED__SIZE>>", 
        "<<DATA_______SIZE>>"
    };
const std::string backTags[6] = 
    {
        "<<ID________FIELD>>",
        "<<FD________FIELD>>",
        "<<POINTER___FIELD>>",
        "<<MEMORY____FIELD>>", 
        "<<MEMORY_____SIZE>>", 
        "<<DATA______FIELD>>"
    };//backward

#define TAG_SIZE 19



struct FFTRequest{
private:
    int windowRadix                 = 10;
    float overlapRate               = 0.0f;
    MAYBE_MEMORY sharedMemoryInfo   = std::nullopt;
    MAYBE_DATA data                 = std::nullopt;
    std::string __mappedID;
    unsigned long long dataLength   = -1;
    void* __memPtr = nullptr;
    int __FD = -1;
public:
    FFTRequest(){}
    FFTRequest(const int& WR, const float& OLR, unsigned long long& counter)
    : windowRadix(WR), overlapRate(OLR)
    {
        __mappedID = std::to_string(counter++);
    };
    BIN Serialize();
    void Deserialize(const BIN& binData);
    void MakeSharedMemory(const SupportedRuntimes& Type, const unsigned long long& dataSize);
    void SetData(std::vector<float>& data);
    MAYBE_DATA getData();
    std::string getID(){return __mappedID;}
};
