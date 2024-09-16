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
    ULL Olength = toOverlapLength(  wp->getDataLength(), 
                                    wp->getOverlapRatio(), 
                                    (1 << wp->getWindowRadix()));
    wp->setOverlapdataLength(Olength);
    
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

        ULL PageSize = adjustToPage<float>(Olength);
        if (ftruncate(__POSIX_FileDes, PageSize) == -1)
        {
            std::cerr << "FD open err: " << errno << " pageSize:"<<Olength << std::endl;
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
    auto OdataLength = mp->getOverlapdataLength();
    auto __memPtr   = reinterpret_cast<void*>(mp->getMemPTR());
    auto __POSIX_FileDes = mp->getPosixFileDes();
    auto sourceSize = mp->getData().size();
    if(sharemem == "ERR")
    {
        return std::nullopt;
    }
    if(sharemem != "")
    {
        std::cout<<"FFTLINUX:89 -- "<<sharemem<<std::endl;
        std::vector<float> result(OdataLength);
        memcpy(result.data(), __memPtr, OdataLength * sizeof(float));
        ULL pageSize = adjustToPage<float>(OdataLength);
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
    ULL OdataLength = mp->getOverlapdataLength();
    if(sharemem == "ERR")
    {
        return std::nullopt;
    }
    
    if(sharemem == "")
    {
        return std::nullopt;
    }
    sharedObj.second = shm_open(sharemem.c_str(), O_RDWR, 0666);
    
    if(sharedObj.second == -1)
    {
        return std::nullopt;
    }
    auto pagedSize = adjustToPage<float>(OdataLength);
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
        auto OdataLength = mw.value().getOverlapdataLength();
        munmap(shobj.first, adjustToPage<float>(OdataLength));
        close(shobj.second);
    }
    else if(mr.has_value())
    {
        auto OdataLength = mr.value().getOverlapdataLength();
        munmap(shobj.first, adjustToPage<float>(OdataLength));
        close(shobj.second);
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
    if(sharemem == "ERR")
    {
        return ;
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