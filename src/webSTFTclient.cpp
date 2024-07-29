#include "webSTFTclient.hpp"
#include <iostream>


void
ClientSTFT::CallbackSet()
{
    
    client.setOnMessageCallback([&](const ix::WebSocketMessagePtr& msg){
        if(msg->type == ix::WebSocketMessageType::Message)
        {
            if(msg->binary)
            {
                std::string bin = msg->str;
                FFTRequest datas(bin);
            }
            if(msg->str == "OK")
            {
                calculateSuccess.set_value(true);
            }
            else
            {
                calculateSuccess.set_value(false);
            }
        }
        if(msg->type == ix::WebSocketMessageType::Close)
        {
            if(msg->closeInfo.code == 0){
                safeDestruct.set_value(true);
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
    if(client.getReadyState() == ix::ReadyState::Open)
    {
        auto okToDestructed = safeDestruct.get_future();
        client.sendText("EOT_SAFE");
        auto exit_status = okToDestructed.wait_for(std::chrono::seconds(5));
        if(exit_status == std::future_status::timeout)
        {
            client.close(1);//err closed
        }
        else
        {
            client.close(0);//safe closed
        }
    }
}

void
ClientSTFT::runtimeFallback()
{

}

DATAF
ClientSTFT::RequestSTFT(DATAF& origin, const int& windowRadix, const float& overlapSize)
{
    
}



int main()
{
    
    std::string bina = "321WR0.366445OL0MD0EDMMEM";
    std::cout<<bina.[bina.find("WR")] << "adsfs" <<std::endl;
    return 0;
}
//     ix::WebSocket webs;
//     webs.setUrl("ws://127.0.0.1:52427/webSTFT");
//     webs.connect(10);
//     webs.sendText("hello");

//     return 0;
// }