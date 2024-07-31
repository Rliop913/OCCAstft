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
ClientSTFT::tryConnect(const PATH& path)
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
    // unsigned long long count = 4412;
    // std::vector<float> tttt(10);
    // for(auto& i : tttt)
    // {
    //     i = 25;
    // }
    // int temp =119;
    // FFTRequest tempRequte(10, 0.7, count);
    // tempRequte.MakeSharedMemory(CUDA, tttt.size());
    // tempRequte.SetData(tttt);
    // tempRequte.__memPtr = &temp;
    // BIN binDATA = tempRequte.Serialize();


    // FFTRequest clonedFFT;
    // clonedFFT.Deserialize(binDATA);
    // std::cout<< clonedFFT.windowRadix << ", " <<
    // clonedFFT.overlapRate << ", " <<
    // clonedFFT.sharedMemoryInfo.value() << ", " <<
    // clonedFFT.__FD << ", " <<
    // *((int*)clonedFFT.__memPtr) << ", " <<
    // clonedFFT.dataLength << ", " <<
    // clonedFFT.getID() << ", " <<
    // std::endl;
    // std::cout<<binDATA<<std::endl;
    // auto output = clonedFFT.getData();
    
    // for(auto i : output.value())
    // {
        
    //     std::cout<< i <<std::endl;
    // }
    // FFTRequest tempRequest;
    // tempRequest.windowRadix = 120;
    // tempRequest.overlapRate =-0.4324;
    // tempRequest.sharedMemoryInfo = "fdasadsfasdffdsf";
    // std::vector<float> vfloat(100);
    // for(int i=0; i<100;i++){
    //     vfloat[i] = i;
    // }
    // tempRequest.data = vfloat;
    // auto result = tempRequest.Serialize();
    // std::cout<<result <<std::endl;

    // FFTRequest copiedTemp;
    // copiedTemp.Deserialize(result);

    // std::cout <<
    // copiedTemp.windowRadix << "," <<
    // copiedTemp.overlapRate << "," <<
    // copiedTemp.sharedMemoryInfo.value() << "," <<
    // copiedTemp.__mappedID <<
    // std::endl;
    // for(auto i : copiedTemp.data.value()){
    //     std::cout << i <<std::endl;
    // }
    // // FallbackList flist;
    // // auto test = flist.itr[0];

    return 0;
}
//     ix::WebSocket webs;
//     webs.setUrl("ws://127.0.0.1:52427/webSTFT");
//     webs.connect(10);
//     webs.sendText("hello");

//     return 0;
// }