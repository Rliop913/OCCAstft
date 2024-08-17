#pragma once
#include <vector>
#include <string>
#include <functional>
#include <future>
#include <optional>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <random>
#include "runtimeChecker.hpp"
#include "FFTStruct.hpp"
#include <IXWebSocket.h>
#include <IXWebSocketServer.h>
#ifdef OS_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>

#endif




// using COSTR             = const std::string;
// using DATAF             = std::vector<float>;
using STRVEC            = std::vector<std::string>;
using ERR_FUNC          = std::function<void(const ix::WebSocketErrorInfo& )>;
using FUTURE_DATA       = std::future<MAYBE_DATA>;
using MAYBE_FUTURE_DATA = std::optional<FUTURE_DATA>;
using PROMISE_DATA      = std::promise<MAYBE_DATA>;
using SHARED_MEMORY     = std::string;





struct STFTProxy{
private:
    //initialized by outside
    FallbackList fallback;
    ERR_FUNC errorHandler;
public:
    ix::WebSocket proxyOBJ;
    int portNumber = -1;
    //states
    SupportedRuntimes gpuType;
    ULL promiseCounter;

    int GeneratePortNumber();
    void RuntimeFallback();
    void SetWebSocketCallback();
    void Disconnect();
    
public:
    std::optional<std::promise<bool>> runnerkilled = std::nullopt;

    std::unordered_map<std::string, PROMISE_DATA> workingPromises;
    std::string STATUS = "OK";

    bool TryConnect(PATH& path);
    bool KillRunner();
    MAYBE_FUTURE_DATA
    RequestSTFT(std::vector<float>& data, 
                const int& windowRadix, 
                const float& overlapRate = 0.5);
    STFTProxy(ERR_FUNC errorCallback, const FallbackList& fbList);
    ~STFTProxy();
    
};
