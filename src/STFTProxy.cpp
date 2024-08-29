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

MAYBE_FUTURE_DATA
STFTProxy::RequestSTFT(std::vector<float>& data, const int& windowRadix, const float& overlapRate)
{
    if(STATUS != "OK")
    {
        return std::nullopt;
    }

    FFTRequest loaded(windowRadix, overlapRate, promiseCounter);
    loaded.MakeSharedMemory(gpuType, data.size());
    loaded.SetData(data);
    
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



int main()
{
    // ULL coutn =0;
    // FFTRequest originTemp(10, 0.5,coutn);
    
    // std::vector<float> testZeroData(100);
    // for(int i=0; i<testZeroData.size(); ++i)
    // {
    //     testZeroData[i] = float(i)/testZeroData.size()+1.0f;
    // }
    // originTemp.MakeSharedMemory(SERVER, testZeroData.size());
    // std::cout<< originTemp.GetSharedMemPath().value();
    // originTemp.SetData(testZeroData);
    // auto result = originTemp.Serialize();

    // FFTRequest cloned(result.value());
    // cloned.MakeSharedMemory(SERVER, testZeroData.size());
    // auto overres = cloned.get_OverlapRate();
    // std::cout << overres << std::endl;
    // auto dat = cloned.FreeAndGetData();
    // for(int i=0; i<100; ++i)
    // {
    //     std::cout << dat.value()[i]<<" , ";
    // }


    // int testData= 9758;
    // int* Dptr = &testData;
    // int handle = 5435;
    // int* hanptr= &handle;


    // capnp::MallocMessageBuilder memField;
    // RequestCapnp::Builder reqObj = memField.initRoot<RequestCapnp>();
    // reqObj.setSharedMemory("TempText");
    // auto data = reqObj.initData(3);
    // data.set(0, 0.5);
    // data.set(1, 54.5);
    // data.set(2, -95.34);
    // reqObj.setMappedID("sample map id");
    // reqObj.setMemPTR(reinterpret_cast<ULL>(Dptr));
    // reqObj.setPosixFileDes(443);
    // reqObj.setWindowsHandlePTR(reinterpret_cast<ULL>(hanptr));
    // reqObj.setWindowRadix(32);
    // reqObj.setOvarlapRate(0.5324);
    // reqObj.setDataLength(42123);
    // std::cout << reqObj.getSharedMemory().cStr()<<std::endl;
    
    // auto flatOut = capnp::messageToFlatArray(memField);
    // auto bytoOut = flatOut.asBytes();
    // std::string binary;
    // binary.resize(bytoOut.size());
    // memcpy(binary.data(), bytoOut.begin(), bytoOut.size());
    // std::cout<< binary<<std::endl;
    
    // kj::ArrayPtr<const capnp::word> readedbyte
    // (
    //     reinterpret_cast<const capnp::word*>(binary.data()),
    //     binary.size() / sizeof(capnp::word)
    // );
    // auto readed_Out = capnp::FlatArrayMessageReader(readedbyte);
    
    // RequestCapnp::Reader redobj = readed_Out.getRoot<RequestCapnp>();
    // auto memptr = reinterpret_cast<int*>(redobj.getMemPTR());
    // auto handleptr = reinterpret_cast<int*>( redobj.getWindowsHandlePTR());
    // std::cout<<
    // redobj.getSharedMemory().cStr() << std::endl<<
    // redobj.getData()[0] << std::endl<<
    // redobj.getMappedID().cStr()<< std::endl<<
    // *memptr<< std::endl<<
    // redobj.getPosixFileDes()<< std::endl<<
    // *handleptr<< std::endl<<
    // redobj.getWindowRadix()<< std::endl<<
    // redobj.getOvarlapRate()<< std::endl<<
    // redobj.getDataLength()<< std::endl;





    FallbackList list;
    list.ServerFallback.push_back("127.0.0.1:54500");
    // list.CUDAFallback.push_back("./cross_gpgpu/CUDA");
    //list.OpenMPFallback.push_back("./cross_gpgpu/OpenMP");
    //list.SerialFallback.push_back("./cross_gpgpu/Serial");
    //list.ServerFallback.push_back("192.168.35.90:54500");
    auto temp = STFTProxy([](const ix::WebSocketErrorInfo& err)
    {
        std::cout<<err.reason << " custom messages"<<std::endl;
        return;
    },list);
    std::vector<float> testZeroData(10000000);
    for(ULL i=0; i<testZeroData.size(); ++i)
    {
        testZeroData[i] = float(i)/testZeroData.size()+1.0f;
    }
    auto promisedData = temp.RequestSTFT(testZeroData, 13, 0.0);

    if(promisedData.has_value())
    {
        auto resOut = promisedData.value().get();
        if(resOut.has_value())
        {
            std::cout<<"got Data!!!"<<std::endl;
            std::cout<<resOut.value()[100]<<std::endl;
        }
    }
    getchar();
    
    temp.proxyOBJ.sendText("CLOSE_REQUEST");
    if(temp.KillRunner())
    {
        std::cout <<"safe closed" << std::endl;
    }
    else
    {
        std::cerr<<"not closed" <<std::endl;
    }
    
    return 0;
}
//     ix::WebSocket webs;
//     webs.setUrl("ws://127.0.0.1:52427/webSTFT");
//     webs.connect(10);
//     webs.sendText("hello");

//     return 0;
// }