#include <vector>
#include <string>

using STRVEC = std::vector<std::string>;
using COSTR = const std::string;

struct ExePath{
    std::string CUDAexe = "./cudaRun";
    std::string OpenCLExe = "./openclRun";
    std::string OpenMPExe = "./openmpRun";
    std::string SerialExe = "./serialRun";
};

struct STFTClient{
private:
    ExePath executePath;
    bool OpenPipe();
    bool ClosePipe();
    bool SendData();
    bool ReceiveData();
public:
    STRVEC CheckRuntimes();
    bool ConnectRuntime(COSTR& runtime);
    STFTClient(const ExePath& paths);
    ~STFTClient();
    
};