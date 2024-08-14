#include "STFTProxy.hpp"
#include <iostream>

STFTProxy::STFTProxy(ERR_FUNC errorCallback, const FallbackList& fbList)
{
    ix::initNetSystem();
    errorHandler = errorCallback;
    fallback = fbList;
    portNumber = GeneratePortNumber();
    proxyOBJ.setPingInterval(45);
    proxyOBJ.enableAutomaticReconnection();
    SetWebSocketCallback();
    RuntimeFallback();
}

int
STFTProxy::GeneratePortNumber()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(49152, 65535);
    int port;
    std::pair<bool, std::string> result;
    while(!result.first)
    {
        port = dis(gen);
        ix::WebSocketServer tempServer(port);
        result = tempServer.listen();
        tempServer.stop();
    }
    return port;
}


void
STFTProxy::SetWebSocketCallback()
{
    proxyOBJ.setOnMessageCallback([&](const ix::WebSocketMessagePtr& msg) {
        if(msg->type == ix::WebSocketMessageType::Message)
        {
            if(msg->binary)
            {
                FFTRequest datas;
                datas.Deserialize(msg->str);
                workingPromises[datas.getID()].set_value(datas);
            }
            else
            {
                if(msg->str == "CLOSE_COMPLETE")
                {
                    if(runnerkilled.has_value())
                    {
                        runnerkilled.value().set_value(true);
                    }
                }
            }
        }
        if(msg->type == ix::WebSocketMessageType::Close)
        {
            if(msg->closeInfo.code == 1000){//safe closed
                proxyOBJ.close(0);
            }
            else{
                RuntimeFallback();
            }
        }
        if(msg->type == ix::WebSocketMessageType::Error)
        {
            errorHandler(msg->errorInfo);
            RuntimeFallback();
        }
    });
}


STFTProxy::~STFTProxy()
{
    ix::uninitNetSystem();
}

void
STFTProxy::RuntimeFallback()
{
    proxyOBJ.close(0);
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

            if(TryConnect(next.value()))
            {
                gpuType = next.value().first;
                break; 
            }
            else
            {
                next = fallback.getNext();
            }
        }
    }
}

bool
STFTProxy::TryConnect(PATH& path)
{
    if (RuntimeCheck::isAvailable(path))
    {
        if(path.first == SupportedRuntimes::SERVER)
        {
            
            proxyOBJ.setUrl(
                "ws://" +
                path.second 
            );
        }
        else
        {
            if (!RuntimeCheck::ExcuteRunner(path.second, portNumber))
            {
                return false;
            }
            proxyOBJ.setUrl(
                "ws://localhost:" +
                std::to_string(portNumber)
            );
            runnerkilled = std::promise<bool>();
        }
        ix::WebSocketInitResult res;
        for(int i=0; i<5; ++i)
        {
            res = proxyOBJ.connect(1);
            if(res.success)
            {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if(!res.success)
        {
            return false;
        }
        proxyOBJ.start();
        return true;
    }
    else
    {
        return false;
    }
}

bool 
STFTProxy::KillRunner()
{
    if(runnerkilled.has_value())
    {
        proxyOBJ.sendText("CLOSE_REQUEST");
        auto waitkill = runnerkilled.value().get_future();
        auto retStatus = waitkill.wait_for(std::chrono::seconds(5));
        if(retStatus == std::future_status::timeout)
        {
            runnerkilled = std::nullopt;
            return false;
        }
        else
        {
            runnerkilled = std::nullopt;
            return true;
        }
    }
    return false;
}

FFTRequest
STFTProxy::LoadToRequest(std::vector<float>& data, const int& windowRadix, const float& overlapRate)
{
    FFTRequest request(windowRadix, overlapRate, promiseCounter);
    request.MakeSharedMemory(gpuType, data.size());
    request.SetData(data);
    return request;
}

MAYBE_FUTURE_DATA
STFTProxy::RequestSTFT(std::vector<float>& data, const int& windowRadix, const float& overlapRate)
{
    if(STATUS != "OK")
    {
        return std::nullopt;
    }
    
    auto loaded = LoadToRequest(data, windowRadix, overlapRate);


    PROMISE_DATA pData;
    workingPromises[loaded.getID()] = std::move(pData);
    proxyOBJ.sendBinary(loaded.Serialize());
    return workingPromises[loaded.getID()].get_future();
}



int main()
{
    FallbackList list;
    list.SerialFallback.push_back("./cross_gpgpu/Serial");
    auto temp = STFTProxy([](const ix::WebSocketErrorInfo& err)
    {
        std::cout<<err.reason<<std::endl;
        return;
    },list);
    std::cout<<temp.STATUS<<std::endl;
    //temp.proxyOBJ.send("CLOSE_REQUEST");
    std::vector<float> testZeroData(10000);




    for(int i=0; i<testZeroData.size(); ++i)
    {
        testZeroData[i] = float(i)/testZeroData.size()+1.0f;
    }
    auto promisedData = temp.RequestSTFT(testZeroData, 10, 0.5);

    if(promisedData.has_value())
    {
        std::cout<<"got future"<<std::endl;
        auto result = promisedData.value().get();
        auto resOut = result.FreeAndGetData();
        if(resOut.has_value())
        {
            std::cout<<"got Data!!!"<<std::endl;
            std::cout<<resOut.value()[100]<<std::endl;
        }
    }
    getchar();
    
    if(temp.KillRunner())
    {
        std::cout <<"safe closed" << std::endl;
    }
    else
    {
        std::cerr<<"not closed" <<std::endl;
    }
    // ULL counter = 4132;
    // std::vector<float> testData(10);
    // for(auto& i : testData)
    // {
    //     i = 3213;
    // }
    // FFTRequest origin(10, 0.5, counter);
    // origin.MakeSharedMemory(CUDA, testData.size());
    // origin.SetData(testData);

    // auto bintest = origin.Serialize();
    // std::cout << bintest <<std::endl;
    // // return 0;
    // FFTRequest cloned;
    // cloned.Deserialize(bintest);
    // auto cloneID = cloned.getID();
    // auto cloneOut = cloned.FreeAndGetData();
    // if(cloneID != origin.getID())
    // {
    //     std::cout << "ID NOT MATCHED" << std::endl;
    // }
    // if(!cloneOut.has_value())
    // {
    //     std::cout << "NO VALUE" << std::endl;
    // }
    // for(int i = 0; i < testData.size();++i)
    // {
    //     std::cout << cloneOut.value()[i] << std::endl;
    //     if(testData[i] != cloneOut.value()[i])
    //     {
    //         std::cout << "IDX: "<< i << "NOT MATCHED. cD: " 
    //         << cloneOut.value()[i] << "originD: " << testData[i] << std::endl;
    //     }
    // }
    return 0;
}
//     ix::WebSocket webs;
//     webs.setUrl("ws://127.0.0.1:52427/webSTFT");
//     webs.connect(10);
//     webs.sendText("hello");

//     return 0;
// }