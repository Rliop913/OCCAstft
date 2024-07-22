#include "STFT.h"


Stft::Stft(CONSTRL mode)
{
    prop["mode"] = mode;
    prop["platform_id"] = 0;
    prop["device_id"] = 0;
    //debug code fragment
    prop["verbose"] = true;
    prop["kernel/verbose"] = true;
    prop["kernel/compiler_flags"] = "-g";
    dev.setup(prop);
}




void
Stft::addNewKernel(CONSTRL kernel_path, CONSTRL kernel_name)
{
    kern[kernel_name] = dev.buildKernel(kernel_path, kernel_name, prop);
}




void
Stft::do_stft()
{

}
