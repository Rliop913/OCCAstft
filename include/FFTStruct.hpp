#pragma once
#include <vector>
#include <optional>
#include <string>
#include <sstream>
#include <thread>

#include "FFTcapnp.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
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
using R             = RequestCapnp::Reader;
using W             = RequestCapnp::Builder;
using MAYBE_BIN     = std::optional<BIN>;
using MAYBE_READ    = std::optional<R>;
using MAYBE_WRITE   = std::optional<W>;
using MAYBE_DATA    = std::optional<std::vector<float>>;
using MAYBE_MEMORY  = std::optional<std::string>;
using MAYBE_SHOBJ   = std::optional<SHMOBJ>;


using CULL          = const unsigned long long;
using ULL           = unsigned long long;




// static const 
// std::string frontTags[3] = 
//     { 
//         "<<WINDOW____RADIX>>", 
//         "<<OVERLAPED__SIZE>>",
//         "<<DATA_____LENGTH>>"
//     };
// static const 
// std::string backTags[6] = 
//     {
//         "<<WINHANDLE_FIELD>>",
//         "<<POSIX_FD__FIELD>>",
//         "<<MEMPOINTERFIELD>>",
//         "<<ID________FIELD>>",
//         "<<MEMORY____FIELD>>", 
//         "<<DATA______FIELD>>"
//     };//backward

#define TAG_SIZE 19



struct FFTRequest{
private:
    capnp::MallocMessageBuilder wField;
    std::unique_ptr<capnp::FlatArrayMessageReader> rField;
    kj::ArrayPtr<const capnp::word> binPtr;
    BIN BinData;
    MAYBE_WRITE mw = std::nullopt;
    MAYBE_READ  mr = std::nullopt;


    // MAYBE_MEMORY sharedMemoryInfo   = std::nullopt;
    // MAYBE_DATA data                 = std::nullopt;
    // std::string __mappedID;
    // void* __memPtr = nullptr;

    // int __POSIX_FileDes = -1;
    // void* __WINDOWS_HANDLEPtr = nullptr;
    // int windowRadix                 = 10;
    // float overlapRate               = 0.0f;
    // ULL dataLength   = -1;
    template<typename T>
    void copyToCapnpParallel(T* dataP, capnp::List<float, capnp::Kind::PRIMITIVE>::Builder* rt, ULL& sourceSize);
    template<typename T>
    void copyToVecParallel(T* dataP, const R* rt, const ULL& sourceSize);
    template<typename T>
    ULL adjustToPage(const ULL& length);

    void Deserialize();
public:
    void MakeWField();
    FFTRequest(const BIN& binary);
    FFTRequest(const int& WR, const float& OLR, ULL& counter);
    // : windowRadix(WR), overlapRate(OLR)
    // {
    //     __mappedID = std::to_string(counter++);
    // };
    MAYBE_BIN Serialize();//will add capnproto
    void MakeSharedMemory(const SupportedRuntimes& Type, CULL& dataSize);
    MAYBE_MEMORY GetSharedMemPath()
    {
        MAYBE_MEMORY sharemem;
        if(mr.has_value())
        {
            sharemem = mr.value().getSharedMemory().cStr();
        }
        else if(mw.has_value())
        {
            sharemem = mw.value().getSharedMemory().cStr();
        }
        if(sharemem == "")
        {
            sharemem = std::nullopt;
        }
        return sharemem;
    }

    //Integrity check from received object
    void StoreErrorMessage();
    bool CheckHasErrorMessage();

    void SetData(std::vector<float>& data);
    [[nodiscard]]
    MAYBE_DATA FreeAndGetData();
    [[nodiscard]]
    MAYBE_DATA GetData();

    MAYBE_SHOBJ GetSHMPtr();
    void FreeSHMPtr(SHMOBJ& shobj);
    //completely unlink
    void FreeData();
    std::string getID()
    {
        std::string mapid;
        if(mw.has_value())
        {
            mapid = mw.value().getMappedID().cStr();
        }
        else if(mr.has_value())
        {
            mapid = mr.value().getMappedID().cStr();
        }
        return mapid;
    }
    int get_WindowRadix()
    {
        int radix;
        if(mw.has_value())
        {
            radix = mw.value().getWindowRadix();
        }
        else if(mr.has_value())
        {
            radix = mr.value().getWindowRadix();
        }
        return radix;
    }
    float get_OverlapRate()
    {
        float oRate;
        if(mw.has_value())
        {
            oRate = mw.value().getOvarlapRate();
        }
        else if(mr.has_value())
        {
            oRate = mr.value().getOvarlapRate();
        }
        return oRate;
    }
    ULL get_dataLength()
    {
        ULL leng;
        if(mw.has_value())
        {
            leng = mw.value().getDataLength();
        }
        else if(mr.has_value())
        {
            leng = mr.value().getDataLength();
        }
        return leng;
    }
    

};


template<typename T>
ULL
FFTRequest::adjustToPage(const ULL& length)
{   
    ULL dataSize = length * sizeof(T);
    unsigned long long PageSize = sysconf(_SC_PAGESIZE);
    return ((dataSize + PageSize - 1) / PageSize) * PageSize;
}


template<typename T>
void
FFTRequest::copyToCapnpParallel(T* dataP, 
                                capnp::List<float, capnp::Kind::PRIMITIVE>::Builder* rt,
                                ULL& sourceSize)
{
    auto lmbd = []( capnp::List<float, capnp::Kind::PRIMITIVE>::Builder* root, 
                    T* dataPTR, 
                    const ULL& sidx, 
                    const ULL& eidx){

        
        T* pidx = dataPTR + sidx;
        for(ULL i = sidx; i< eidx; ++i)
        {
            root->set(i, *pidx);
            ++pidx;
        }
    };
    
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    ULL chunkSize = sourceSize / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        ULL startIdx = i * chunkSize;
        ULL endIdx = (i == numThreads - 1) ? sourceSize : (i + 1) * chunkSize; // 마지막 스레드는 남은 모든 데이터를 처리

        threads.emplace_back(lmbd, rt, dataP, startIdx, endIdx);
    }

    for (auto& t : threads) {
        t.join();
    }
}


template<typename T>
void
FFTRequest::copyToVecParallel(T* dataP, const R* rt, const ULL& sourceSize)
{
    auto lmbd = []( const R* root, 
                    T* dataPTR, 
                    const ULL& sidx, 
                    const ULL& eidx){

        auto source = root->getData();
        T* pidx = dataPTR + sidx;
        for(ULL i = sidx; i< eidx; ++i)
        {

            *pidx = source[i];
            ++pidx;
        }
    };
    
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    ULL chunkSize = sourceSize / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        ULL startIdx = i * chunkSize;
        ULL endIdx = (i == numThreads - 1) ? sourceSize : (i + 1) * chunkSize; // 마지막 스레드는 남은 모든 데이터를 처리

        threads.emplace_back(lmbd, rt, dataP, startIdx, endIdx);
    }

    for (auto& t : threads) {
        t.join();
    }
}