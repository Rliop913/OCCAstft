#pragma once
#include <vector>
#include <string>
#include <functional>
#include <future>
#include <optional>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include "runtimeChecker.hpp"
#include "FFTStruct.hpp"
#include <IXWebSocket.h>

#ifdef OS_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>

#endif



#define FIXED_PORT 52437

// using COSTR             = const std::string;
// using DATAF             = std::vector<float>;
using STRVEC            = std::vector<std::string>;
using ERR_FUNC          = std::function<void(const ix::WebSocketErrorInfo& )>;
using FUTURE_DATA       = std::future<FFTRequest>;
using MAYBE_FUTURE_DATA = std::optional<FUTURE_DATA>;
using PROMISE_DATA      = std::promise<FFTRequest>;
using SHARED_MEMORY     = std::string;





struct STFTProxy{
private:
    //initialized by outside
    FallbackList fallback;
    ERR_FUNC errorHandler;
    ix::WebSocket proxyOBJ;

    //states
    SupportedRuntimes gpuType;
    ULL promiseCounter;

    MAYBE_RUNNER runnerHandle = std::nullopt;
    std::promise<bool>* runnerKilled = nullptr;

    
    void RuntimeFallback();
    void SetWebSocketCallback();
    FFTRequest LoadToRequest(std::vector<float>& data, const int& windowRadix, const float& overlapRate);
    void Disconnect();
    
public:

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