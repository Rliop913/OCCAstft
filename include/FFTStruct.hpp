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
using SHMOBJ        = std::pair<void*, int>;


#endif
#ifdef OS_WINDOWS
#include <Windows.h>
using SHMOBJ        = std::pair<LPVOID, HANDLE>;
#endif
#include "runtimeChecker.hpp"


using BIN           = std::string;

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

#define TAG_SIZE 19



struct FFTRequest{
private:

    capnp::MallocMessageBuilder wField;
    std::unique_ptr<capnp::FlatArrayMessageReader> rField;

    kj::ArrayPtr<const capnp::word> binPtr;

    BIN BinData;
    MAYBE_WRITE mw = std::nullopt;
    MAYBE_READ  mr = std::nullopt;

    template<typename T>
    void copyToCapnpParallel
    (
        T* dataP, 
        capnp::List<float, capnp::Kind::PRIMITIVE>::Builder* rt, 
        ULL& sourceSize
    );
    
    template<typename T>
    void copyToVecParallel
    (
        T* dataP, 
        const R* rt, 
        const ULL& sourceSize
    );
    
    template<typename T>
    ULL adjustToPage(const ULL& length);

    void Deserialize();

public:

    //construct as Reader
    FFTRequest(const BIN& binary);

    //construct as Writer
    FFTRequest(const int& WR, const float& OLR, ULL& counter);

    //serialize with capnproto
    MAYBE_BIN Serialize();

    //generates new shared memory.
    void MakeSharedMemory(const SupportedRuntimes& Type, CULL& dataSize);

    //make Writer object for making binary
    void MakeWField();

    //stores error message
    void StoreErrorMessage();
    //checks error message exists
    bool CheckHasErrorMessage();

    //close shared memory pointer. it doesn't destroy shared memory.
    void FreeSHMPtr(SHMOBJ& shobj);

    MAYBE_MEMORY GetSharedMemPath();
    std::string getID();
    int get_WindowRadix();
    float get_OverlapRate();
    ULL get_dataLength();
    
    ULL get_OverlapDataLength();
    ULL toOverlapLength(const ULL& dataLength, 
                        const float& overlapRatio, 
                        const ULL& windowSize);

    //gets SharedMemory pointer from shmem path. The memory should be allocated.
    MAYBE_SHOBJ GetSHMPtr();

    //Destroy Shared memory if exists and return data
    [[nodiscard]]
    MAYBE_DATA FreeAndGetData();

    //just get vector data.
    [[nodiscard]]
    MAYBE_DATA GetData();

    //destroy shared memory
    void FreeData();
    //Load vector data to shared memory or capnp Field(Binary send)
    void SetData(std::vector<float>& data);

    void SetOption(const std::string& options);
    std::string GetOption();
};

#ifdef OS_POSIX

template<typename T>
ULL
FFTRequest::adjustToPage(const ULL& length)
{   
    ULL dataSize = length * sizeof(T);
    unsigned long long PageSize = sysconf(_SC_PAGESIZE);
    return ((dataSize + PageSize - 1) / PageSize) * PageSize;
}

#else

template<typename T>
ULL
FFTRequest::adjustToPage(const ULL& length)
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    DWORD pageSize  = sysInfo.dwPageSize;
    ULL DSize = length;
    DSize = ((DSize + pageSize - 1) / pageSize) * pageSize;
    DSize *= sizeof(T);
    return DSize;
}


#endif

template<typename T>
void
FFTRequest::copyToCapnpParallel(T* dataP, 
                                capnp::List<float, capnp::Kind::PRIMITIVE>::Builder* rt,
                                ULL& sourceSize)
{
    auto lmbd = []( capnp::List<float, capnp::Kind::PRIMITIVE>::Builder* root, 
                    T* dataPTR, 
                    const ULL& sidx, 
                    const ULL& eidx)
    {
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