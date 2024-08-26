#pragma once
#include "FFTStruct.hpp"
#ifdef OS_POSIX
void
FFTRequest::MakeSharedMemory(const SupportedRuntimes& Type, const ULL& dataSize)
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

        std::cout<<
        "SharememFree>> isMapFree: "<< freemap << std::endl <<
        "isFileDes Free: " << freefd << std::endl <<
        "isShareMemLink Free: " << freelink << std::endl <<
        "FS_Linux:96" << std::endl;

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
    ULL dataLength = mp->getDataLength();

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


// MAYBE_DATA
// FFTRequest::GetData()
// {
//     if(!mr.has_value())
//     {
//         return std::nullopt;
//     }
//     auto mp = &mr.value();
//     std::string sharemem = mp->getSharedMemory().cStr();
//     auto dataLength = mp->getDataLength();
//     auto __memPtr = reinterpret_cast<void*>(mp->getMemPTR());
//     auto sourceSize = mp->getData().size();
//     if(sharemem != "")
//     {
//         std::vector<float> result(dataLength);
//         memcpy(result.data(), __memPtr, dataLength * sizeof(float));
//         return std::move(result);
//     }
//     else if(sourceSize != 0)
//     {
//         std::vector<float> result(sourceSize);
//         copyToVecParallel(result.data(), mp, sourceSize);
//         std::cout << "got data FS_Linux:187 "<< result[150] <<std::endl;
//         return std::move(result);
//     }
//     else
//     {
//         return std::nullopt;
//     }
// }

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