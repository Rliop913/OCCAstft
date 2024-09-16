#pragma once
#include "FFTStruct.hpp"
#ifdef OS_WINDOWS



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
        ULL PageSize = adjustToPage<float>(Olength);
        auto
        __WINDOWS_HANDLEPtr = CreateFileMapping
        (
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            static_cast<DWORD>( (PageSize >> 32) & 0xFFFFFFFF),
            static_cast<DWORD>(PageSize & 0xFFFFFFFF),
            fullpath.c_str()
        );
        
        if (__WINDOWS_HANDLEPtr == NULL)
        {

            std::cerr << 
            "handle open err" <<
            GetLastError() << std::endl;
            wp->setSharedMemory("");
            return;
        }

        auto __memPtr = MapViewOfFile
        (
            __WINDOWS_HANDLEPtr,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            PageSize
        );

        if (__memPtr == NULL)
        {
            std::cerr <<
            "mmap err" << 
            errno << std::endl;

            CloseHandle(__WINDOWS_HANDLEPtr);
            __WINDOWS_HANDLEPtr = NULL;
            wp->setSharedMemory("");
            return;
        }

        wp->setWindowsHandlePTR(reinterpret_cast<ULL>(__WINDOWS_HANDLEPtr));
        wp->setMemPTR(reinterpret_cast<ULL>(__memPtr));
    }
    return;
}

MAYBE_DATA
FFTRequest::FreeAndGetData()
{
    auto result = GetData();
    FreeData();
    return result;
}


void
FFTRequest::FreeData()
{
    void* __memPtr;
    void* WIN_HANDLE;
    std::string sharemem;
    if(mw.has_value())
    {
        auto pw = &mw.value();
        __memPtr = reinterpret_cast<void*>(pw->getMemPTR());
        WIN_HANDLE = reinterpret_cast<void*>(pw->getWindowsHandlePTR());
        sharemem = pw->getSharedMemory().cStr();
    }
    else if(mr.has_value())
    {
        auto pw = &mr.value();
        __memPtr = reinterpret_cast<void*>(pw->getMemPTR());
        WIN_HANDLE = reinterpret_cast<void*>(pw->getWindowsHandlePTR());
        sharemem = pw->getSharedMemory().cStr();
    }
    if(sharemem == "ERR")
    {
        return std::nullopt;
    }
    if(sharemem != "")
    {
        
        if(!UnmapViewOfFile(__memPtr))
        {
            std::cerr << "FW:135 unmapviewoffile failed" << std::endl;
        }
        if(!CloseHandle(WIN_HANDLE))
        {
            std::cerr << "FW:139 closehandle failed" << std::endl;
        }
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
    if(sharemem == "")
    {
        return std::nullopt;
    }
    if(sharemem == "ERR")
    {
        return std::nullopt;
    }
    sharedObj.second = OpenFileMapping
    (
        FILE_MAP_ALL_ACCESS,
        FALSE,
        sharemem.c_str()
    );
    if(sharedObj.second == NULL)
    {
        std::cerr << "FW:165 can't open file mapping for reading" << std::endl;
        return std::nullopt;
    }
    auto pagedSize = adjustToPage<float>(OdataLength);
    sharedObj.first = MapViewOfFile
    (
        sharedObj.second,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        pagedSize
    );
    if(sharedObj.first == NULL)
    {
        std::cerr << "FW:180 can't map view of file" << std::endl;
        CloseHandle(sharedObj.second);
        return std::nullopt;
    }
    return sharedObj;
}

void
FFTRequest::FreeSHMPtr(SHMOBJ& shobj)
{
    if(!UnmapViewOfFile(shobj.first))
    {
        std::cerr << "FW:194 unmapviewoffile failed" << std::endl;
    }
    if(!CloseHandle(shobj.second))
    {
        std::cerr << "FW:198 closehandle failed" << std::endl;
    }
}
#endif