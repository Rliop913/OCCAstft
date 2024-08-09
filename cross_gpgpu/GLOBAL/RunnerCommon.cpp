#include "RunnerInterface.hpp"




Runner::Runner()
{
    InitEnv();
    BuildKernel();
    ServerInit();
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
        delete server;
    }
}

void
Runner::ServerInit()
{
    if(server != nullptr)
    {
        delete server;
    }
    server = new ix::WebSocketServer(FIXED_PORT);//need to add random
    auto err = server->listen();
    if(!err.first)
    {
        ServerInit();
    }
}

int main()
{
    Runner mainOBJ = Runner();
    mainOBJ.server->start();
    mainOBJ.server->wait();
    return 0;
}