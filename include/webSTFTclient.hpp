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

#define FIXED_PORT 52437

using STRVEC        = std::vector<std::string>;
using COSTR         = const std::string;
using DATAF         = std::vector<float>;
using ERR_FUNC      = std::function<void(const ix::WebSocketErrorInfo& )>;
using FUTURE_DATA   = std::future<FFTRequest>;
using MAYBE_FUTURE_DATA = std::optional<FUTURE_DATA>;
using PROMISE_DATA  = std::promise<FFTRequest>;


using SHARED_MEMORY = std::string;

struct ClientSTFT{
private:
    
    ix::WebSocket client;
    FallbackList fallback;
    ERR_FUNC errorHandler;
    SupportedRuntimes supportingType;

    void runtimeFallback();
    void CallbackSet();
    FFTRequest LoadToRequest(const FFTRequest& request);
    SHARED_MEMORY AllocSharedMemory(const FFTRequest& request);
    bool FreeSharedMemory(const FFTRequest& request);

    std::promise<bool>* serverkilled = nullptr;
public:
    std::unordered_map<int, PROMISE_DATA> workingPromises;
    std::string STATUS = "OK";
    void killServer();
    bool tryConnect(const PATH& path);

    MAYBE_FUTURE_DATA
    RequestSTFT(FFTRequest& request );

    ClientSTFT(ERR_FUNC errorCallback, const FallbackList& fbList);
    ~ClientSTFT();
    
};