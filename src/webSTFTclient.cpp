#include "webSTFTclient.hpp"
#include <iostream>

ClientSTFT::ClientSTFT(ERR_FUNC errorCallback, const FallbackList& fbList)
{
    errorHandler = errorCallback;
    fallback = fbList;
    runtimeFallback();
}



void
ClientSTFT::CallbackSet()
{
    client.setOnMessageCallback([&](const ix::WebSocketMessagePtr& msg) {
        if(msg->type == ix::WebSocketMessageType::Message)
        {
            if(msg->binary)
            {
                FFTRequest datas;
                datas.Deserialize(msg->str);
                workingPromises[datas.getID()].set_value(datas);
            }
        }
        if(msg->type == ix::WebSocketMessageType::Close)
        {
            if(msg->closeInfo.code == 0){
                if(serverkilled != nullptr)
                {
                    serverkilled->set_value(true);
                }
            }
            else{
                runtimeFallback();
            }
        }
        if(msg->type == ix::WebSocketMessageType::Error)
        {
            errorHandler(msg->errorInfo);
            runtimeFallback();
        }
    });
}


ClientSTFT::~ClientSTFT()
{
    client.close(0);
}

void
ClientSTFT::runtimeFallback()
{
    client.close(0);
    auto next = fallback.getNext();
    while(true)
    {
        if(!next.has_value())
        {
            STATUS = "Run Out Fallback";
            break;
        }
        else
        {
            if (RuntimeCheck::isAvailable(next.value()))
            {
                if(tryConnect(next.value()))
                {
                    supportingType = next.value().first;
                    break; 
                }
            }
            else
            {
                next = fallback.getNext();
            }
        }
    }
}

bool
ClientSTFT::tryConnect(PATH& path)
{
    if (RuntimeCheck::isAvailable(path))
    {
        if(path.first == SupportedRuntimes::SERVER)
        {
            
            client.setUrl(
                "ws://" +
                path.second +
                ":" + 
                std::to_string(FIXED_PORT) +
                "/webSTFT"
            );
        }
        else
        {
            running_process = RuntimeCheck::ExcuteRunner(path.second);
            if (!running_process.has_value())
            {
                return false;
            }
            client.setUrl(
                "ws://127.0.0.1:" +
                std::to_string(FIXED_PORT)+
                "/localSTFT"
            );
        }
        auto res = client.connect(5);
        if(!res.success)
        {
            return false;
        }
        return true;
    }
    else
    {
        return false;
    }
}

FFTRequest
ClientSTFT::LoadToRequest(std::vector<float>& data, const int& windowRadix, const float& overlapRate)
{
    FFTRequest request(windowRadix, overlapRate, mapCounter);
    request.MakeSharedMemory(supportingType, data.size());
    request.SetData(data);
    return request;
}

MAYBE_FUTURE_DATA
ClientSTFT::RequestSTFT(std::vector<float>& data, const int& windowRadix, const float& overlapRate)
{
    if(STATUS != "OK")
    {
        return std::nullopt;
    }
    
    auto loaded = LoadToRequest(data, windowRadix, overlapRate);


    PROMISE_DATA pData;
    workingPromises[loaded.getID()] = std::move(pData);
    client.sendBinary(loaded.Serialize());
    return workingPromises[loaded.getID()].get_future();
}



int main()
{
    ULL counter = 4132;
    std::vector<float> testData(10);
    for(auto& i : testData)
    {
        i = 3213;
    }
    FFTRequest origin(10, 0.5, counter);
    origin.MakeSharedMemory(CUDA, testData.size());
    origin.SetData(testData);

    auto bintest = origin.Serialize();
    std::cout << bintest <<std::endl;
    // return 0;
    FFTRequest cloned;
    cloned.Deserialize(bintest);
    auto cloneID = cloned.getID();
    auto cloneOut = cloned.getData();
    if(cloneID != origin.getID())
    {
        std::cout << "ID NOT MATCHED" << std::endl;
    }
    if(!cloneOut.has_value())
    {
        std::cout << "NO VALUE" << std::endl;
    }
    for(int i = 0; i < testData.size();++i)
    {
        std::cout << cloneOut.value()[i] << std::endl;
        if(testData[i] != cloneOut.value()[i])
        {
            std::cout << "IDX: "<< i << "NOT MATCHED. cD: " 
            << cloneOut.value()[i] << "originD: " << testData[i] << std::endl;
        }
    }
    return 0;
}
//     ix::WebSocket webs;
//     webs.setUrl("ws://127.0.0.1:52427/webSTFT");
//     webs.connect(10);
//     webs.sendText("hello");

//     return 0;
// }