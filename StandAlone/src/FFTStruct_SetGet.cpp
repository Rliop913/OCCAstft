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
FFTRequest::get_WindowSizeEXP()
{
    int exp = 0;
    if(mw.has_value())
    {
        exp = mw.value().getWindowRadix();
    }
    else if(mr.has_value())
    {
        exp = mr.value().getWindowRadix();
    }
    return exp;
}

float
FFTRequest::get_OverlapRate()
{
    float oRate = 0.0;
    if(mw.has_value())
    {
        oRate = mw.value().getOverlapRatio();
    }
    else if(mr.has_value())
    {
        oRate = mr.value().getOverlapRatio();
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


ULL
FFTRequest::get_OverlapDataLength()
{
    ULL leng = 0;
    if(mw.has_value())
    {
        leng = mw.value().getOverlapdataLength();
    }
    else if(mr.has_value())
    {
        leng = mr.value().getOverlapdataLength();
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

void
FFTRequest::SetOption(const std::string& options)
{
    if(!mw.has_value())
    {
        return;
    }
    auto mp = &mw.value();
    mp->setOptions(options);
}

std::string
FFTRequest::GetOption()
{
    if(mr.has_value())
    {
        return mr.value().getOptions().cStr();
    }
    else if(mw.has_value())
    {
        return mw.value().getOptions().asString();
    }
    else
    {
        return "ERR NO OBJECT";
    }
}