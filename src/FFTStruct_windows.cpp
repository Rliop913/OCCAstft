#pragma once
#include "FFTStruct.hpp"
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