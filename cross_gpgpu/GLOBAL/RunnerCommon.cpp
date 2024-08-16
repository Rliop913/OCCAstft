#include "RunnerInterface.hpp"




Runner::Runner(const int& portNumber)
{
    std::cout<<"RC:INIT"<<std::endl;
    InitEnv();
    std::cout<<"RC:9"<<std::endl;
    BuildKernel();
    std::cout<<"RC:11"<<std::endl;
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
        std::cout<<"got mempath RC:40"<< mempath.value() <<std::endl;
        dataInfo = req.GetSHMPtr();
        std::cout << "RC:44 gotshmptr"<<std::endl;
        if(!dataInfo.has_value())
        {
            std::cout << "RC:47 nullopt"<<std::endl;
            return std::nullopt;
        }
        else
        {
            VECF accResult(req.get_dataLength());
            std::cout << "RC:52 memcpy"<<std::endl;
            memcpy( accResult.data(), 
                    dataInfo.value().first, 
                    req.get_dataLength() * sizeof(float));
                    std::cout << "RC:55 memcpy"<<std::endl;
            return std::move(accResult);
        }
        
    }
    else
    {
        std::cout<<"got data RC:58"<<std::endl;
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
                        
                        FFTRequest received(msg->str);
                        received.MakeWField();
                        auto data = AccessData(received);
                        if(data.has_value())
                        {
                            auto result =
                            ActivateSTFT(   data.value(), 
                                            received.get_WindowRadix(),
                                            received.get_OverlapRate());
                            if(dataInfo.has_value() && result.has_value())
                            {
                                memcpy( dataInfo.value().first, 
                                        result.value().data(),
                                        received.get_dataLength() * sizeof(float));
                            }
                            else if(result.has_value())
                            {
                            std::cout<<"calced RC:90"<<std::endl;
                                received.SetData(result.value());
                            }
                            else
                            {
                                received.StoreErrorMessage();
                            }
                            if(dataInfo.has_value())
                            {
                                received.FreeSHMPtr(dataInfo.value());
                            }
                        }
                        else//error message
                        {
                            received.StoreErrorMessage();
                        }
                        auto serialResult = received.Serialize();
                        if(serialResult.has_value())
                        {
                            webSocket.sendBinary(serialResult.value());
                        }
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
    std::cout<<"RC:169"<<std::endl;
    mainOBJ.server->wait();
    std::cout<<"end Runner"<<std::endl;
    ix::uninitNetSystem();
    return 0;
}