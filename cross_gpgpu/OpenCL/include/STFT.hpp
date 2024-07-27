#include <CL/opencl.hpp>


struct codes{
    Kernel rmDC;
    Kernel overlapNWindow;
    Kernel bitReverse;
    Kernel endProcess;
    Kernel butterfly;
    Kernel toPower;
};

struct STFT{
private:
    void InitEnv();

    CL_env env;
    codes kens;
public:
    STFT();
    ~STFT();
};