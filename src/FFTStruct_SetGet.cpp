#include "FFTStruct.hpp"

std::string 
FFTRequest::getID()
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

int 
FFTRequest::get_WindowRadix()
{
    int radix = 0;
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

float
FFTRequest::get_OverlapRate()
{
    float oRate = 0.0;
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

ULL
FFTRequest::get_dataLength()
{
    ULL leng = 0;
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

MAYBE_MEMORY
FFTRequest::GetSharedMemPath()
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