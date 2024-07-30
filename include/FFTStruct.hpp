#pragma once
#include <vector>
#include <optional>
#include <string>

using BIN           = std::string;
using MAYBE_DATA    = std::optional<std::vector<float>>;
using MAYBE_MEMORY  = std::optional<std::string>;

const std::string frontTags[3] = 
    { 
        "<<WINDOW____RADIX>>", 
        "<<OVERLAPED__SIZE>>", 
        "<<DATA______FIELD>>"
    };
const std::string backTags[3] = 
    {
        "<<END______MEMORY>>", 
        "<<MEMORY____FIELD>>", 
        "<<END________DATA>>"
    };//backward

#define TAG_SIZE 19



struct FFTRequest{
public:
    int windowRadix                 = 10;
    float overlapSize               = 0.0f;
    MAYBE_MEMORY sharedMemoryInfo   = std::nullopt;
    MAYBE_DATA data                 = std::nullopt;
private:
    int __mappedID;
public:
    int& GetMappedID(){return __mappedID;}
    BIN Serialize();
    void Deserialize(const BIN& binData);
};



BIN
FFTRequest::Serialize()
{
    BIN wrBIN   = std::to_string(windowRadix);
    BIN osBIN   = std::to_string(overlapSize);
    BIN dataSIZE;
    BIN dataBIN ;
    BIN memSIZE ;
    BIN memBIN  ;
    BIN idBIN   = std::to_string(__mappedID);
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
        int intSize = sharedMemoryInfo.value().size();

        memSIZE = std::to_string(intSize);

        memBIN = BIN
        (   
            reinterpret_cast<char*>( sharedMemoryInfo.value().data() ), 
            intSize
        );
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
    dataBIN     + backTags[2]   +
    memSIZE     + backTags[1]   +
    memBIN      + backTags[0]   +
    idBIN;

    return Serial;
}


void
FFTRequest::Deserialize(const BIN& binData )
{
    size_t tagPos[6];
    
    for (int i = 0; i < 3; ++i) 
    {
        auto pos = binData.find( frontTags[i] );

        if (pos == std::string::npos ) 
        {
            goto ERR_DIVE;
        }

        tagPos[i] = pos;
    }


    for (int i = 0; i < 3; ++i)
    {
        auto pos = binData.rfind( backTags[i] );

        if (pos == std::string::npos) 
        {
            goto ERR_DIVE;
        }

        tagPos[5 - i] = pos;
    }


    auto binBeg = binData.begin();

    auto bin_WR = binBeg + tagPos[0] + TAG_SIZE;
    auto bin_OL = binBeg + tagPos[1] + TAG_SIZE;
    auto bin_MD = binBeg + tagPos[2] + TAG_SIZE;
    auto bin_ED = binBeg + tagPos[3] + TAG_SIZE;
    auto bin_MM = binBeg + tagPos[4] + TAG_SIZE;
    auto bin_EM = binBeg + tagPos[5] + TAG_SIZE;

    std::string wrTemp              (binBeg, bin_WR - TAG_SIZE);
    std::string osTemp              (bin_WR, bin_OL - TAG_SIZE);
    std::string DataContained       (bin_OL, bin_MD - TAG_SIZE);
    std::string SharedMemContained  (bin_ED, bin_MM - TAG_SIZE);
    std::string idArea              (bin_EM, binData.end());
    try
    {
        windowRadix = std::stoi(wrTemp);
        overlapSize = std::stof(osTemp);
        __mappedID  = std::stoi(idArea);
    } 
    catch (const std::exception &e) 
    { 
        goto ERR_DIVE; 
    }

    if (DataContained == "0") 
    { 
        data = std::nullopt;
    } 
    else 
    {
        int vectorSize = 0;
        try
        { 
            vectorSize = std::stoi(DataContained); 
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

    if (SharedMemContained == "0") 
    { 
        sharedMemoryInfo = std::nullopt; 
    } 
    else 
    { 
        sharedMemoryInfo = std::string( bin_MM, bin_EM - TAG_SIZE ); 
    }


    return;//Safe return

    ERR_DIVE: // Error Dive
    windowRadix         = -1;
    overlapSize         = 0.0f;
    data                = std::nullopt;
    sharedMemoryInfo    = std::nullopt;
    __mappedID          = -1;
    return;//Err return
}
