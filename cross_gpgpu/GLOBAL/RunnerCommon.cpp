#include "RunnerInterface.hpp"




Runner::Runner(const int& portNumber)
{
    InitEnv();
    BuildKernel();
    if(!ServerInit(portNumber))
    {
        std::cerr<<"ERR on server init"<<std::endl;
    }
}


Runner::~Runner()
{
    if(env != nullptr)
    {
        delete env;
    }
    if(kens != nullptr)
    {
        delete kens;
    }
    if(server != nullptr)
    {
        server->stop();
        delete server;
    }
}

MAYBE_DATA
Runner::AccessData(FFTRequest& req)
{
    auto mempath = req.GetSharedMemPath();
    if(mempath.has_value())
    {
        dataInfo = req.GetSHMPtr();
        if(!dataInfo.has_value())
        {
            return std::nullopt;
        }
        else
        {
            VECF accResult(req.dataLength);
            memcpy( accResult.data(), 
                    dataInfo.value().first, 
                    req.dataLength * sizeof(float));
            return std::move(accResult);
        }
        
    }
    else
    {
        return std::move(req.GetData());
    }
}

bool
Runner::ServerInit(const int& pNum)
{
    if(server != nullptr)
    {
        delete server;
    }
    server = new ix::WebSocketServer(pNum, "0.0.0.0");//need to add random

    server->setOnClientMessageCallback([&](
            std::shared_ptr<ix::ConnectionState> connectionState,
            ix::WebSocket &webSocket,
            const ix::WebSocketMessagePtr& msg)
            {
                if(msg->type == ix::WebSocketMessageType::Open)
                {
                    std::cout<<"Open Transmission"<<std::endl;
                }
                if(msg->type == ix::WebSocketMessageType::Message)
                {
                    if(msg->binary)
                    {
                        FFTRequest received;
                        received.Deserialize(msg->str);
                        auto data = AccessData(received);
                        if(data.has_value())
                        {
                            auto result =
                            ActivateSTFT(   data.value(), 
                                            received.windowRadix,
                                            received.overlapRate);
                            if(dataInfo.has_value() && result.has_value())
                            {
                                memcpy( dataInfo.value().first, 
                                        result.value().data(),
                                        received.dataLength * sizeof(float));
                            }
                            else if(result.has_value())
                            {
                                received.SetData(result.value());
                            }
                            else
                            {
                                received.BreakIntegrity();
                            }
                            if(dataInfo.has_value())
                            {
                                received.FreeSHMPtr(dataInfo.value());
                            }
                        }
                        else//error message
                        {
                            received.BreakIntegrity();
                        }
                        
                        webSocket.sendBinary(received.Serialize());
                    }
                    else
                    {
                        if(msg->str == "CLOSE_REQUEST")
                        {
                            webSocket.sendText("CLOSE_COMPLETE");
                            std::thread([&]() {
                            server->stop();
                            }).detach();
                            
                        }
                    }
                }
            }
        );
    auto err = server->listen();
    if(!err.first)
    {
        return false;
    }
    server->start();
    return true;
}

int main(int argc, char *argv[])
{
    int portNumber = 54500; //default portnumber
    if(argc == 2)
    {
        try
        {
            portNumber = std::stoi(argv[1]);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }
    std::cout<<"runner started" << portNumber <<std::endl;
    ix::initNetSystem();
    Runner mainOBJ = Runner(portNumber);
    mainOBJ.server->wait();
    std::cout<<"end Runner"<<std::endl;
    ix::uninitNetSystem();
    return 0;
}