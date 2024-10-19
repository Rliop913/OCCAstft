#include "STFTProxy.hpp"
#include <iostream>

STFTProxy::STFTProxy(ERR_FUNC errorCallback, const FallbackList& fbList)
{
    ix::initNetSystem();
    errorHandler = errorCallback;
    fallback = fbList;
    portNumber = GeneratePortNumber();
    ix::WebSocketPerMessageDeflateOptions options(true, false, false, 15, 15);
    proxyOBJ.setPerMessageDeflateOptions(options);
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
                FFTRequest datas(msg->str);
                workingPromises[datas.getID()].set_value(datas.FreeAndGetData());
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
    KillRunner(true);
    ix::uninitNetSystem();
}

void
STFTProxy::RuntimeFallback()
{
    proxyOBJ.close(0);
    auto next = fallback.getNext();
    while(STATUS == "OK")
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
                "ws://127.0.0.1:" +
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
STFTProxy::KillRunner(bool noFallbackAnyMore)
{
    if(noFallbackAnyMore)
    {
        STATUS = "NOFALLBACK";
    }
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

MAYBE_FUTURE_DATA
STFTProxy::RequestSTFT( std::vector<float>& data, 
                        const int& windowSizeEXP, 
                        const float& overlapRate,
                        const std::string& options)
{
    if(STATUS != "OK")
    {
        return std::nullopt;
    }

    FFTRequest loaded(windowSizeEXP, overlapRate, promiseCounter);
    loaded.MakeSharedMemory(gpuType, data.size());
    loaded.SetData(data);
    loaded.SetOption(options);
    
    PROMISE_DATA pData;
    workingPromises[loaded.getID()] = std::move(pData);
    auto serializeResult = loaded.Serialize();
    if(!serializeResult.has_value())
    {
        return std::nullopt;
    }
    
    proxyOBJ.sendBinary(loaded.Serialize().value());
    return workingPromises[loaded.getID()].get_future();
}
