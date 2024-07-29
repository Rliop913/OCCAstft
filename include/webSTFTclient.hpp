#pragma once
#include <vector>
#include <string>
#include <functional>
#include <future>
#include <optional>
#include <runtimeChecker.hpp>

// #include <IXWebSocket.h>
#include <IXWebSocket.h>

using STRVEC = std::vector<std::string>;
using COSTR = const std::string;
using DATAF = std::vector<float>;
using ERR_FUNC = std::function<void(const ix::WebSocketErrorInfo&)>;

using MAYBE_DATA = std::optional<std::vector<float>>;
using MAYBE_MEMORY = std::optional<std::string>;
struct FFTRequest{
    int windowRadix = 10;
    float overlapSize = 0.0f;
    MAYBE_MEMORY sharedMemoryInfo;
    MAYBE_DATA data;
    FFTRequest(const std::string& binData){
        binData.find("WR");
    }
};

//Calculation fallback lists.
struct FallbackList{
    std::string CUDAExe = "SKIP";//first check
    std::string OpenCLExe = "SKIP";//second ckeck
    std::string OpenMPExe = "SKIP";//...
    std::string ServerURL = "SKIP";//server url or ip
    std::string SerialExe = "SKIP";//last. it should be ok
};


struct ClientSTFT{
private:
    bool ISLOCAL = true;
    ix::WebSocket client;
    std::promise<bool> calculateSuccess;
    ERR_FUNC errorHandler;
    void runtimeFallback();
    void CallbackSet();
    void Init();
    

public:
    std::promise<bool> safeDestruct;
    bool tryConnect();
    DATAF
    RequestSTFT(DATAF& origin, const int& windowRadix, const float& overlapSize);
    ClientSTFT(ERR_FUNC errorCallback, const FallbackList& fbList);
    ~ClientSTFT();
    
};