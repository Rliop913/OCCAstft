#include "FFTStruct.hpp"

void
FFTRequest::MakeSharedMemory(const SupportedRuntimes& Type, const unsigned long long& dataSize)
{
    dataLength = (unsigned long long)dataSize;
    if(Type != SERVER)
    {
        sharedMemoryInfo = "/STFT" + __mappedID + "SHAREDMEM";

        __FD = shm_open(sharedMemoryInfo.value().c_str(), O_CREAT | O_RDWR, 0666);
        
        if (__FD == -1)
        {
            std::cerr << "shm open err" << std::endl;
            sharedMemoryInfo = std::nullopt;
            return;
        }

        unsigned long long PageSize = sysconf(dataSize * sizeof(float));

        if (ftruncate(__FD, PageSize) == -1)
        {
            std::cerr << "FD open err" << std::endl;
            shm_unlink(sharedMemoryInfo.value().c_str());
            sharedMemoryInfo = std::nullopt;
            return;
        }
        
        __memPtr = mmap
            (
                0, 
                PageSize, 
                PROT_READ | PROT_WRITE,
                MAP_SHARED, 
                __FD, 
                0
            );
        
        if (__memPtr == MAP_FAILED)
        {
            
            std::cerr << "mmap err" << __memPtr << std::endl;
            close(__FD);
            shm_unlink(sharedMemoryInfo.value().c_str());
            sharedMemoryInfo = std::nullopt;
            return;
        }
    }
    return;
}

void
FFTRequest::SetData(std::vector<float>& requestedData)
{
    if(sharedMemoryInfo.has_value())
    {
        memcpy(__memPtr, requestedData.data(), dataLength * sizeof(float));
    }
    else
    {
        data = std::move(requestedData);
    }
}


MAYBE_DATA
FFTRequest::getData()
{
    if(sharedMemoryInfo.has_value())
    {
        std::vector<float> result(dataLength);
        memcpy(result.data(), __memPtr, dataLength * sizeof(float));
        unsigned long long pageSize = sysconf(dataLength * sizeof(float));
        int freemap = munmap(__memPtr, pageSize);
        int freefd  = close(__FD);
        int freelink= shm_unlink(sharedMemoryInfo.value().c_str());
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




BIN
FFTRequest::Serialize()
{
    BIN wrBIN   = std::to_string(windowRadix);
    BIN osBIN   = std::to_string(overlapRate);
    BIN dataSIZE;
    BIN dataBIN ;
    BIN memSIZE = std::to_string(dataLength);//Shared Memory Length
    BIN memBIN  ;
    BIN ptrBIN  = std::to_string(reinterpret_cast<uintptr_t>(__memPtr));
    BIN FDBIN   = std::to_string(__FD);
    BIN idBIN   = __mappedID;
    if (data.has_value() ) 
    {
        int intSize = data.value().size();

        dataSIZE = std::to_string(intSize);
        
        dataBIN = BIN
        (  
            reinterpret_cast<char*>(data.value().data() ),
            intSize * sizeof(float)
        );
    } 
    else 
    {
        dataSIZE    = "0";
        dataBIN     = "";
    }

    if( sharedMemoryInfo.has_value() ) 
    {
        memBIN = sharedMemoryInfo.value();
    } 
    else
    {
        memSIZE = "0";
        memBIN  = "";
    }

    BIN Serial = 
    wrBIN       + frontTags[0]  +
    osBIN       + frontTags[1]  +
    dataSIZE    + frontTags[2]  +
    dataBIN     + backTags[5]   +
    memSIZE     + backTags[4]   +
    memBIN      + backTags[3]   +
    ptrBIN      + backTags[2]   +
    FDBIN       + backTags[1]   +
    idBIN       + backTags[0];

    return Serial;
}


void
FFTRequest::Deserialize(const BIN& binData )
{
    if(false)
    {
        ERR_DIVE: // Error Dive
        windowRadix         = -1;
        overlapRate         = 0.0f;
        data                = std::nullopt;
        sharedMemoryInfo    = std::nullopt;
        __mappedID          = -1;
        return;
    }
    size_t tagPos[9];
    
    for (int i = 0; i < 3; ++i) 
    {
        auto pos = binData.find( frontTags[i] );

        if (pos == std::string::npos ) 
        {
            goto ERR_DIVE;
        }

        tagPos[i] = pos;
    }


    for (int i = 0; i < 6; ++i)
    {
        auto pos = binData.rfind( backTags[i] );

        if (pos == std::string::npos) 
        {
            goto ERR_DIVE;
        }

        tagPos[8 - i] = pos;
    }


    auto binBeg = binData.begin();

    auto bin_WR = binBeg + tagPos[0] + TAG_SIZE;
    auto bin_OL = binBeg + tagPos[1] + TAG_SIZE;
    auto bin_DS = binBeg + tagPos[2] + TAG_SIZE;
    auto bin_DF = binBeg + tagPos[3] + TAG_SIZE;
    auto bin_MS = binBeg + tagPos[4] + TAG_SIZE;
    auto bin_MF = binBeg + tagPos[5] + TAG_SIZE;
    auto bin_Ptr= binBeg + tagPos[6] + TAG_SIZE;
    auto bin_FD = binBeg + tagPos[7] + TAG_SIZE;
    auto bin_ID = binBeg + tagPos[8] + TAG_SIZE;



    BIN wrTemp              (binBeg, bin_WR - TAG_SIZE);
    BIN osTemp              (bin_WR, bin_OL - TAG_SIZE);
    BIN dataSize            (bin_OL, bin_DS - TAG_SIZE);
    BIN dataField           (bin_DS, bin_DF - TAG_SIZE);
    BIN dataSize_ForSharemem(bin_DF, bin_MS - TAG_SIZE);
    BIN sharememString      (bin_MS, bin_MF - TAG_SIZE);
    BIN sharePTR            (bin_MF, bin_Ptr- TAG_SIZE);
    BIN fileDes             (bin_Ptr,bin_FD - TAG_SIZE);
    __mappedID = std::string(bin_FD, bin_ID - TAG_SIZE);


    try
    {
        windowRadix = std::stoi(wrTemp);
        overlapRate = std::stof(osTemp);
        __FD     = std::stoi(fileDes);
        dataLength  = std::stoul(dataSize_ForSharemem);
        
        __memPtr = reinterpret_cast<void*>(std::stoul(sharePTR));
    } 
    catch (const std::exception &e) 
    { 
        goto ERR_DIVE; 
    }

    if (dataSize == "0") 
    { 
        data = std::nullopt;
    } 
    else 
    {
        int vectorSize = 0;
        try
        { 
            vectorSize = std::stoi(dataSize); 
        }
        catch (const std::exception &e) 
        { 
            goto ERR_DIVE;
        }

        data = std::vector<float>(vectorSize);

        memcpy
        ( 
            data.value().data(), 
            &binData[ tagPos[2] + TAG_SIZE ], 
            vectorSize * sizeof(float)
        );
    }

    if (dataSize_ForSharemem == "0") 
    { 
        sharedMemoryInfo = std::nullopt; 
    } 
    else 
    { 
        sharedMemoryInfo = sharememString;
    }


    return;//Safe return

}
