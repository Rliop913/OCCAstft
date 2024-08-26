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
        std::cout<<"Size: "<< sizeof(capnp::word)<<std::endl;
        
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
    
    std::cout<< BinData.size()<<", "<< binPtr.size()<<std::endl;
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
        std::cout << "got data FS_133 "<< result[150] <<std::endl;
        return std::move(result);
    }
    else
    {
        return std::nullopt;
    }
}