#include "FFTStruct.hpp"
#include <iostream>
#ifdef OS_POSIX
void
FFTRequest::MakeSharedMemory(const SupportedRuntimes& Type, const unsigned long long& dataSize)
{
    if(!mw.has_value())
    {
        return;
    }
    auto wp = &mw.value();
    wp->setDataLength(dataSize);
    std::string mappedID = wp->getMappedID().cStr();
    std::string fullpath = "/STFT" + mappedID + "SHAREDMEM";

    if(Type != SERVER)
    {
        wp->setSharedMemory(fullpath);

        auto __POSIX_FileDes = shm_open(fullpath.c_str(), O_CREAT | O_RDWR, 0666);
        
        if (__POSIX_FileDes == -1)
        {
            std::cerr << "shm open err" << std::endl;
            wp->setSharedMemory("");
            return;
        }

        ULL PageSize = adjustToPage<float>(dataSize);
        if (ftruncate(__POSIX_FileDes, PageSize) == -1)
        {
            std::cerr << "FD open err: " << errno << " pageSize:"<<dataSize << std::endl;
            shm_unlink(fullpath.c_str());
            wp->setSharedMemory("");
            return;
        }
        
        auto __memPtr = mmap
            (
                0, 
                PageSize, 
                PROT_READ | PROT_WRITE,
                MAP_SHARED, 
                __POSIX_FileDes, 
                0
            );
        
        if (__memPtr == MAP_FAILED)
        {
            
            std::cerr << "mmap err" << __memPtr << std::endl;
            close(__POSIX_FileDes);
            shm_unlink(fullpath.c_str());
            wp->setSharedMemory("");
            return;
        }
        wp->setPosixFileDes(__POSIX_FileDes);
        wp->setMemPTR(reinterpret_cast<ULL>(__memPtr));
    }
    return;
}


MAYBE_DATA
FFTRequest::FreeAndGetData()
{
    if(!mr.has_value())
    {
        return std::nullopt;
    }
    auto mp = &mr.value();
    std::string sharemem = "";
    if(mp->hasSharedMemory())
    {
        
        sharemem =mp->getSharedMemory().cStr();

    }
    auto dataLength = mp->getDataLength();
    auto __memPtr   = reinterpret_cast<void*>(mp->getMemPTR());
    auto __POSIX_FileDes = mp->getPosixFileDes();
    auto sourceSize = mp->getData().size();
    
    if(sharemem != "")
    {
        std::vector<float> result(dataLength);
        memcpy(result.data(), __memPtr, dataLength * sizeof(float));
        ULL pageSize = adjustToPage<float>(dataLength);
        int freemap = munmap(__memPtr, pageSize);
        int freefd  = close(__POSIX_FileDes);
        int freelink= shm_unlink(sharemem.c_str());
        return std::move(result);
    }
    else if(sourceSize != 0)
    {
        std::vector<float> result(sourceSize);
        copyToVecParallel(result.data(), mp, sourceSize);
        return std::move(result);
    }
    else
    {
        return std::nullopt;
    }
}

MAYBE_SHOBJ
FFTRequest::GetSHMPtr()
{
    if(!mr.has_value())
    {
        return std::nullopt;
    }
    auto mp = &mr.value();
    SHMOBJ sharedObj;
    std::string sharemem = mp->getSharedMemory().cStr();
    auto dataLength = mp->getDataLength();
    if(sharemem == "")
    {
        return std::nullopt;
    }
    sharedObj.second = shm_open(sharemem.c_str(), O_RDWR, 0666);
    if(sharedObj.second == -1)
    {
        return std::nullopt;
    }
    auto pagedSize = adjustToPage<float>(dataLength);
    sharedObj.first = mmap( 0, 
                            pagedSize, 
                            PROT_READ | PROT_WRITE, 
                            MAP_SHARED, 
                            sharedObj.second, 
                            0);
    if(sharedObj.first == MAP_FAILED)
    {
        close(sharedObj.second);
        return std::nullopt;
    }
    return sharedObj;
}

void
FFTRequest::FreeSHMPtr(SHMOBJ& shobj)
{
    if(mw.has_value())
    {
        auto dataLength = mw.value().getDataLength();
        munmap(shobj.first, adjustToPage<float>(dataLength));
        close(shobj.second);
    }
    else if(mr.has_value())
    {
        auto dataLength = mr.value().getDataLength();
        munmap(shobj.first, adjustToPage<float>(dataLength));
        close(shobj.second);
    }
}


MAYBE_DATA
FFTRequest::GetData()
{
    if(!mr.has_value())
    {
        return std::nullopt;
    }
    auto mp = &mr.value();
    std::string sharemem = mp->getSharedMemory().cStr();
    auto dataLength = mp->getDataLength();
    auto __memPtr = reinterpret_cast<void*>(mp->getMemPTR());
    auto sourceSize = mp->getData().size();
    if(sharemem != "")
    {
        std::vector<float> result(dataLength);
        memcpy(result.data(), __memPtr, dataLength * sizeof(float));
        return std::move(result);
    }
    else if(sourceSize != 0)
    {
        std::vector<float> result(sourceSize);
        copyToVecParallel(result.data(), mp, sourceSize);
        std::cout << "got data FS:179 "<< result[150] <<std::endl;
        return std::move(result);
    }
    else
    {
        return std::nullopt;
    }
}

void
FFTRequest::FreeData()
{
    ULL dataLength;
    void* __memPtr;
    int __POSIX_FileDes;
    std::string sharemem;
    if(mw.has_value())
    {
        auto pw = &mw.value();
        dataLength = pw->getDataLength();
        __memPtr = reinterpret_cast<void*>(pw->getMemPTR());
        __POSIX_FileDes = pw->getPosixFileDes();
        sharemem = pw->getSharedMemory().cStr();
    }
    else if(mr.has_value())
    {
        auto pw = &mr.value();
        dataLength = pw->getDataLength();
        __memPtr = reinterpret_cast<void*>(pw->getMemPTR());
        __POSIX_FileDes = pw->getPosixFileDes();
        sharemem = pw->getSharedMemory().cStr();

    }
    if(sharemem != "")
    {
        ULL pageSize = adjustToPage<float>(dataLength);
        int freemap = munmap(__memPtr, pageSize);
        int freefd  = close(__POSIX_FileDes);
        int freelink= shm_unlink(sharemem.c_str());
    }
}


#endif
#ifdef OS_WINDOWS

ULL
FFTRequest::adjustToPage()
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    DWORD pageSize  = sysInfo.dwPageSize;
    ULL DSize = dataLength * sizeof(float);
    DSize = ((DSize + pageSize - 1) / pageSize) * pageSize;
    return DSize;
}

void
FFTRequest::MakeSharedMemory(const SupportedRuntimes& Type, CULL& dataSize)
{
    dataLength = (ULL)dataSize;
    if(Type != SERVER)
    {
        auto DSize = adjustToPage();
        sharedMemoryInfo = "/STFT" + __mappedID + "SHAREDMEM";

        __WINDOWS_HANDLEPtr = CreateFileMapping
        (
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            static_cast<DWORD>( (DSize >> 32) & 0xFFFFFFFF),
            static_cast<DWORD>(DSize & 0xFFFFFFFF),
            sharedMemoryInfo.value().c_str()
        );
        
        if (__WINDOWS_HANDLEPtr == NULL)
        {

            std::cerr << 
            "handle open err" <<
            GetLastError() << std::endl;
            
            sharedMemoryInfo = std::nullopt;
            return;
        }

        __memPtr = MapViewOfFile
        (
            __WINDOWS_HANDLEPtr,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            DSize
        );

        if (__memPtr == NULL)
        {
            std::cerr <<
            "mmap err" << 
            errno << std::endl;

            CloseHandle(__WINDOWS_HANDLEPtr);
            __WINDOWS_HANDLEPtr = NULL;

            sharedMemoryInfo = std::nullopt;
            return;
        }
    }
    return;
}



MAYBE_DATA
FFTRequest::getData()
{
    if(sharedMemoryInfo.has_value())
    {
        std::vector<float> result(dataLength);
        memcpy(result.data(), __memPtr, dataLength * sizeof(float));

        UnmapViewOfFile(__memPtr);
        CloseHandle(__WINDOWS_HANDLEPtr);
        return std::move(result);
    }
    else if(data.has_value())
    {
        return std::move(data);
    }
    else
    {
        return std::nullopt;
    }
}



#endif

void
FFTRequest::SetData(std::vector<float>& requestedData)
{
    if(!mw.has_value())
    {
        return;
    }
    auto mp = &mw.value();
    std::string sharemem = mp->getSharedMemory().cStr();
    void* __memPtr = reinterpret_cast<void*>( mp->getMemPTR());
    ULL dataLength = requestedData.size();
    mp->setDataLength(dataLength);
    if(sharemem != "")
    {
        memcpy(__memPtr, requestedData.data(), dataLength * sizeof(float));
    }
    else
    {
        auto dataP = mp->initData(dataLength);
        copyToCapnpParallel(requestedData.data(), &dataP, dataLength);
    }
}


void
FFTRequest::StoreErrorMessage()
{
    if(mw.has_value())
    {
        mw.value().setSharedMemory("ERR");
    }
}

bool
FFTRequest::CheckHasErrorMessage()
{
    if(mr.has_value())
    {
        std::string sharemem = mr.value().getSharedMemory().cStr();
        if(sharemem == "ERR")
        {
            return false;
        }
    }
    return true;
}


MAYBE_BIN
FFTRequest::Serialize()
{
    if(mw.has_value())
    {
        auto serialized = capnp::messageToFlatArray(wField);
        BIN binOut;
        binOut.resize(serialized.size() * sizeof(capnp::word));
        memcpy(binOut.data(), serialized.begin(), serialized.size() * sizeof(capnp::word));
        std::cout<<"Size: "<< sizeof(capnp::word)<<std::endl;
        
        return std::move(binOut);
    }
    return std::nullopt;

    // std::vector<BIN> DBin(9);
    // DBin[0]     = std::to_string(windowRadix);
    // DBin[1]     = std::to_string(overlapRate);
    // DBin[2]     = std::to_string(dataLength);
    // DBin[3]     = "";//data
    // DBin[4]     = "";//memory
    // DBin[5]     = __mappedID;
    // DBin[6]     = std::to_string(reinterpret_cast<uintptr_t>(__memPtr));
    // DBin[7]     = std::to_string(__POSIX_FileDes);
    // DBin[8]    = std::to_string(reinterpret_cast<uintptr_t>(__WINDOWS_HANDLEPtr));
    
    // if (data.has_value()) 
    // {
    //     DBin[3] = BIN
    //     (  
    //         reinterpret_cast<char*>(data.value().data()),
    //         dataLength * sizeof(float)
    //     );
    // } 
    // if(sharedMemoryInfo.has_value()) 
    // {
    //     DBin[4] = sharedMemoryInfo.value();
    // } 
    
    // BIN Serial = 
    // DBin[0]     + frontTags[0]  +
    // DBin[1]     + frontTags[1]  +
    // DBin[2]     + frontTags[2]  +
    // DBin[3]     + backTags[5]   +
    // DBin[4]     + backTags[4]   +
    // DBin[5]     + backTags[3]   +
    // DBin[6]     + backTags[2]   +
    // DBin[7]     + backTags[1]   +
    // DBin[8]     + backTags[0]   ;
    // return Serial;
}


void
FFTRequest::Deserialize()
{
    binPtr
    = kj::ArrayPtr<const capnp::word>
    (
        reinterpret_cast<const capnp::word*>(BinData.data()),
        BinData.size() / sizeof(capnp::word)
    );
    
    std::cout<< BinData.size()<<", "<< binPtr.size()<<std::endl;
    rField = std::make_unique<capnp::FlatArrayMessageReader>(binPtr);
    // rField = capnp::FlatArrayMessageReader(binRead);
    mr = rField->getRoot<RequestCapnp>();
    // if(false)
    // {
    //     ERR_DIVE: // Error Dive
    //     windowRadix         = -1;
    //     overlapRate         = 0.0f;
    //     data                = std::nullopt;
    //     sharedMemoryInfo    = std::nullopt;
    //     __mappedID          = -1;
    //     return;
    // }
    // size_t tagPos[9];
    // for (int i = 0; i < 3; ++i) 
    // {
    //     auto pos = binData.find( frontTags[i] );
    //     if (pos == std::string::npos ) 
    //     {
    //         goto ERR_DIVE;
    //     }
    //     tagPos[i] = pos;
    // }

    // for (int i = 0; i < 6; ++i)
    // {
    //     auto pos = binData.rfind( backTags[i] );
    //     if (pos == std::string::npos) 
    //     {
    //         goto ERR_DIVE;
    //     }
    //     tagPos[8 - i] = pos;
    // }


    // auto binBeg = binData.begin();
    // std::vector<std::string::const_iterator> position(11);
    // position[0]     = binBeg + tagPos[0]    + TAG_SIZE;
    // position[1]     = binBeg + tagPos[1]    + TAG_SIZE;
    // position[2]     = binBeg + tagPos[2]    + TAG_SIZE;
    // position[3]     = binBeg + tagPos[3]    + TAG_SIZE;
    // position[4]     = binBeg + tagPos[4]    + TAG_SIZE;
    // position[5]     = binBeg + tagPos[5]    + TAG_SIZE;
    // position[6]     = binBeg + tagPos[6]    + TAG_SIZE;
    // position[7]     = binBeg + tagPos[7]    + TAG_SIZE;
    // position[8]     = binBeg + tagPos[8]    + TAG_SIZE;


    // BIN winRad              (binBeg,        position[0]     - TAG_SIZE);
    // BIN overRate            (position[0],   position[1]     - TAG_SIZE);
    // BIN dataL               (position[1],   position[2]     - TAG_SIZE);
    // BIN dataBin             (position[2],   position[3]     - TAG_SIZE);
    // BIN memPath             (position[3],   position[4]     - TAG_SIZE);
    // __mappedID = std::string(position[4],   position[5]     - TAG_SIZE);
    // BIN memAdd              (position[5],   position[6]     - TAG_SIZE);
    // BIN PosixFD             (position[6],   position[7]     - TAG_SIZE);
    // BIN WinHand             (position[7],   position[8]     - TAG_SIZE);

    // try
    // {
    //     windowRadix         = std::stoi(winRad);
    //     overlapRate         = std::stof(overRate);
    //     dataLength          = std::stoul(dataL);
    //     __POSIX_FileDes     = std::stoi(PosixFD);
    //     __memPtr            = reinterpret_cast<void*>(std::stoull(memAdd));
    //     __WINDOWS_HANDLEPtr = reinterpret_cast<void*>(std::stoull(WinHand));
    // } 
    // catch (const std::exception &e) { goto ERR_DIVE; }

    // if (dataBin == "") { data = std::nullopt; } 
    // else 
    // {
    //     data = std::vector<float>(dataLength);
    //     memcpy
    //     ( 
    //         data.value().data(), 
    //         dataBin.data(), 
    //         dataLength * sizeof(float)
    //     );
    // }

    // if (memPath == "") { sharedMemoryInfo = std::nullopt; } 
    // else { sharedMemoryInfo = memPath; }

    // return;//Safe return

}


FFTRequest::FFTRequest(const BIN& binary)
:BinData(std::move(binary))
{
    Deserialize();
}

FFTRequest::FFTRequest(const int& WR, const float& OLR, ULL& mapCounter)
{

    mw = wField.initRoot<RequestCapnp>();
    auto pw = &mw.value();
    
    // writeTemp = writeTemp.memField.initRoot<RequestCapnp>();
    pw->setWindowRadix(WR);
    pw->setOvarlapRate(OLR);
    pw->setMappedID(std::to_string(mapCounter));
    pw->setSharedMemory("");
    pw->setMemPTR(0);
    pw->setPosixFileDes(-1);
    pw->setWindowsHandlePTR(-1);
    pw->setDataLength(0);
    
}

void
FFTRequest::MakeWField()
{
    mw = wField.initRoot<RequestCapnp>();
    auto pw = &mw.value();
    auto pr = &mr.value();
    pw->setWindowRadix(pr->getWindowRadix());
    pw->setOvarlapRate(pr->getOvarlapRate());
    pw->setMappedID(pr->getMappedID().cStr());
    pw->setSharedMemory(pr->getSharedMemory().cStr());
    pw->setMemPTR(pr->getMemPTR());
    pw->setPosixFileDes(pr->getPosixFileDes());
    pw->setWindowsHandlePTR(pr->getWindowsHandlePTR());
    pw->setDataLength(pr->getDataLength());
}