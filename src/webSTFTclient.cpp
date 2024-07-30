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
                workingPromises[datas.GetMappedID()].set_value(datas);
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
    return true;
}

FFTRequest
ClientSTFT::LoadToRequest(const FFTRequest& request)
{
    FFTRequest temp;
    return temp;
}

MAYBE_FUTURE_DATA
ClientSTFT::RequestSTFT(FFTRequest& request)
{
    if(STATUS != "OK")
    {
        return std::nullopt;
    }
    
    auto loaded = LoadToRequest(request);
    PROMISE_DATA pData;
    workingPromises[loaded.GetMappedID()] = std::move(pData);
    client.sendBinary(loaded.Serialize());
    return workingPromises[loaded.GetMappedID()].get_future();
}



int main()
{
    FFTRequest tempRequest;
    tempRequest.windowRadix = 120;
    tempRequest.overlapSize =-0.4324;
    tempRequest.sharedMemoryInfo = "fdasadsfasdffdsf";
    std::vector<float> vfloat(100);
    for(int i=0; i<100;i++){
        vfloat[i] = i;
    }
    tempRequest.data = vfloat;
    auto result = tempRequest.Serialize();
    std::cout<<result <<std::endl;

    FFTRequest copiedTemp;
    copiedTemp.Deserialize(result);

    std::cout <<
    copiedTemp.windowRadix << "," <<
    copiedTemp.overlapSize << "," <<
    copiedTemp.sharedMemoryInfo.value() << "," <<
    copiedTemp.GetMappedID() <<
    std::endl;
    for(auto i : copiedTemp.data.value()){
        std::cout << i <<std::endl;
    }
    // FallbackList flist;
    // auto test = flist.itr[0];

    return 0;
}
//     ix::WebSocket webs;
//     webs.setUrl("ws://127.0.0.1:52427/webSTFT");
//     webs.connect(10);
//     webs.sendText("hello");

//     return 0;
// }