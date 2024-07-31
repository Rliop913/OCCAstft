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





struct ClientSTFT{
private:

    ix::WebSocket client;
    FallbackList fallback;
    ERR_FUNC errorHandler;
    SupportedRuntimes supportingType;
    unsigned long long mapCounter;
    void runtimeFallback();
    void CallbackSet();
    FFTRequest LoadToRequest(std::vector<float>& data, const int& windowRadix, const float& overlapRate);
    std::promise<bool>* serverkilled = nullptr;
    void killServer();
    
public:
    std::unordered_map<std::string, PROMISE_DATA> workingPromises;
    std::string STATUS = "OK";
    bool tryConnect(const PATH& path);

    MAYBE_FUTURE_DATA
    RequestSTFT(std::vector<float>& data, const int& windowRadix, const float& overlapRate = 0.5);

    ClientSTFT(ERR_FUNC errorCallback, const FallbackList& fbList);
    ~ClientSTFT();
    
};