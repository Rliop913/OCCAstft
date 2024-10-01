#include "FFTStruct.hpp"
#include <iostream>
#ifdef OS_POSIX

#include "FFTStruct_linux.cpp"

#endif
#ifdef OS_WINDOWS

#include "FFTStruct_windows.cpp"

#endif


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
        return std::move(binOut);
    }
    return std::nullopt;
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
    capnp::ReaderOptions options;
    options.traversalLimitInWords = std::numeric_limits<ULL>::max();
    rField = std::make_unique<capnp::FlatArrayMessageReader>(binPtr, options);
    mr = rField->getRoot<RequestCapnp>();
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
    
    pw->setWindowRadix(WR);
    pw->setOverlapRatio(OLR);
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
    pw->setOverlapRatio(pr->getOverlapRatio());
    pw->setMappedID(pr->getMappedID().cStr());
    pw->setSharedMemory(pr->getSharedMemory().cStr());
    pw->setMemPTR(pr->getMemPTR());
    pw->setPosixFileDes(pr->getPosixFileDes());
    pw->setWindowsHandlePTR(pr->getWindowsHandlePTR());
    pw->setDataLength(pr->getDataLength());
    pw->setOverlapdataLength(pr->getOverlapdataLength());
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
    if(sharemem == "ERR")
    {
        return std::nullopt;
    }
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
        return std::move(result);
    }
    else
    {
        return std::nullopt;
    }
}

ULL
FFTRequest::toOverlapLength(const ULL& dataLength, 
                            const float& overlapRatio, 
                            const ULL& windowSize)
{
    ULL quot = 0;
    if(overlapRatio == 0.0f){
        quot = dataLength / windowSize + 1;
    }
    else{
        quot = ((dataLength ) / (windowSize * (1.0f - overlapRatio))) + 1;
    }
    return quot * windowSize;
}