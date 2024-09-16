
# GPGPU Batch STFT Project

## Overview

This project is a multi-platform (GPGPU) supported implementation of the Short-Time Fourier Transform (STFT). It is designed to perform STFT operations efficiently across various parallel computing environments such as CUDA, OpenCL, OpenMP, and Serial. The project ensures stable execution across diverse systems and deployment environments, making it a reliable and scalable solution for signal processing, particularly for time-frequency analysis of audio and speech signals.

Additionally, the project allows users to request calculations from a server via WebSocket connections, providing flexibility in deployment and usage scenarios.

## Performance

Tested on NVIDIA H100, Ubuntu, CUDA 12.3

![windowSize 64](ProfileResults/Figure/64.png)
![windowSize 128](ProfileResults/Figure/128.png)
![windowSize 256](ProfileResults/Figure/256.png)
![windowSize 512](ProfileResults/Figure/512.png)
![windowSize 1024](ProfileResults/Figure/1024.png)
![windowSize 2048](ProfileResults/Figure/2048.png)
![windowSize 4096](ProfileResults/Figure/4096.png)
![windowSize 8192](ProfileResults/Figure/8192.png)
![windowSize 16384](ProfileResults/Figure/16384.png)
![windowSize 32768](ProfileResults/Figure/32768.png)
![windowSize 65536](ProfileResults/Figure/65536.png)
![windowSize 131072](ProfileResults/Figure/131072.png)
![windowSize 262144](ProfileResults/Figure/262144.png)
![windowSize 524288](ProfileResults/Figure/524288.png)
![windowSize 1048576](ProfileResults/Figure/1048576.png)

## Key Features

- **Platform Support**: Supports STFT operations in CUDA, OpenCL, OpenMP, and Serial environments.
- **Modular Codebase**: Provides a modular code structure for each platform, ensuring ease of maintenance and expansion.
- **IXWebSocket-Based Communication**: Manages external communication and asynchronous processing through WebSocket using the IXWebSocket library. Users can request computations from a server through WebSocket connections.
- **OCCA Integration**: Leverages OCCA (Open Concurrent Compute Abstraction) to easily convert and execute kernel code across various parallel computing environments.

## Directory Structure

- **cross_gpgpu/**: Contains modular implementations for CUDA, OpenCL, OpenMP, and Serial platforms.
  - **GLOBAL/**: default codes for runners.
  - **CUDA/**: GPGPU implementation based on CUDA.
  - **HIP/**: Not implemented yet.
  - **METAL/**: Not implemented yet.
  - **OpenCL/**: GPGPU implementation based on OpenCL.
  - **OpenMP/**: CPU parallel processing implementation based on OpenMP.
  - **Serial/**: Sequential processing implementation without parallelism.
  - **RunnerTemplate/**: Contains template implementations for adding new vendors.
  - **occaprofileRunner/**: A special Runner implementation for performance testing of this implementation. runs with custom fallback.
  - **testRunner/**: Special Runner implementation to extract data from clfft and cufft. runs with custom fallback.
- **include/**: Header files and utility code used across platforms.
- **src/**: Core implementation files for STFT and related functionalities.
- **kernel_build.sh**: Script to generate kernel code for various platforms using OCCA.
- **capnpsetter.sh**: Script to set capnproto.
- **nvccPtxBuild.sh**: Script to build ptx file from .cu file.
## Installation and Build

### Requirements

- CMake 3.5 or higher
- CUDA Toolkit (optional for CUDA builds)
- OpenCL library (optional for OpenCL builds)
- OpenMP-supported compiler
- CapnProto (library for data serialization and deserialization)
- IXWebSocket (library for WebSocket communication)

### Build Procedure

1. Clone the repository.

   ```bash
   git clone https://github.com/Rliop913/GPGPU_Batch_STFT.git
   cd GPGPU_Batch_STFT
   ```

2. Generate the build using CMake.

   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```
   NOTE: Perhaps your first build will fail related to ixwebsocket. Build it again and it will be resolved.

3. Once the build is complete, executables for each vendor will be generated under the `cross_build/` directory.

   For example, `cross_build/CUDA/cudaRun.exe`, `cross_build/OpenCL/openclRun.exe`, etc.

## Usage

The program processes input signal data through STFT and returns the results. You can execute the built files for each platform as a WebSocket server, providing the port number as an argument.

### Example

1. **Running the Executable**

   To run the WebSocket server using the built executable, use the following command:

   ```bash
   ./cross_build/CUDA/cudaRun.exe 54500
   ```

   This command runs the `cudaRun.exe` program and starts the WebSocket server on port `54500`.

2. **Using the STFTProxy Object**

   The following is an example of using the `STFTProxy` object to manage the executable and set up WebSocket communication:

   ```cpp
   #include "STFTProxy.hpp"
   #include <iostream>
   #include <vector>

   int main() {
       // Add expected executable paths to FallbackList and set server information
       FallbackList list;
       list.OpenMPFallback.push_back("./cross_gpgpu/OpenMP");
       list.SerialFallback.push_back("./cross_gpgpu/Serial");
       list.ServerFallback.push_back("127.0.0.1:54500");
       list.CustomFallback.push_back("./path_to_your_custom_runner/yourRunner.exe");//NOTE: It works without problem even without the exe extension.

       // Define WebSocket error handling function
       auto temp = STFTProxy([](const ix::WebSocketErrorInfo& err)
       {
           std::cout << err.reason << std::endl;
       }, list);

       // Create test data
       std::vector<float> testZeroData(10000);
       for (int i = 0; i < testZeroData.size(); ++i)
       {
           testZeroData[i] = float(i) / testZeroData.size() + 1.0f;
       }

       // Request STFT operation
       auto promisedData = temp.RequestSTFT(testZeroData, 10, 0.5 , "--hanning_window --half_complex_return");
   
       // RequestSTFT( FloatVector, WindowRadix, OverlapRate, Options)
       // FloatVector: float vector
       // WindowRadix: Radix data of WindowSize. 2 ^ WindowRadix
       // OverlapRate: overlap rate. 0.6 means 60% overlap
       // Options: options for preprocess, returned shape.
   
       if (promisedData.has_value()) // checks promise
       {
           auto resOut = promisedData.value().get(); // get data from promise
           if (resOut.has_value()) // checks data from promise.
           {
               // Use the result
           }
       }

       getchar();

       // Safely close the running Runner (only works if the proxy object directly started the Runner)
       // Note: If the proxy starts the Runner directly, the proxy and Runner can use shared memory between processes, providing faster execution.
       if (temp.KillRunner())
       {
           std::cout << "safe closed" << std::endl;
       }
       else
       {
           std::cerr << "not closed" << std::endl;
       }

       return 0;
   }
   ```

   This code demonstrates how to manage executables and set up WebSocket communication for STFT processing. The `KillRunner` function works **only if the proxy object directly started the Runner** and requests a safe shutdown. Additionally, if the proxy starts the Runner directly, the proxy and Runner can use shared memory between processes, enabling faster execution.
## Available Options
- **PreProcess**

    **--hanning_window**: hanning window

    **--hamming_window**: hamming window

    **--blackman_window**: blackman window

    **--nuttall_window** : nuttall window

    **--blackman_nuttall_window** : As written

    **--blackman_harris_window** : As written

    **--flattop_window** : As written

    **--gaussian_window=(sigmaSize)<<sigma**: gaussian window. Enter the sigma value to use it.
  
    e.g. --gaussian_window=4.312345<<sigma

    **--remove_dc**: removes the DC component within the window.

- **PostProcess**

    **--half_complex_return**:  leverages the symmetry of the FFT output to remove the redundant second half of the complex result. Since memory and communication primarily use float arrays, this option packs the real and imaginary components alternately (real, imaginary, real, imaginary) in the available space, thereby reducing memory usage and ensuring compatibility."

    **no postprocess option mentioned(default)**: returns the squared (power) value.

## Customizing

### Customize a Runner

If you want to add a new vendor or other implementation to this project, please follow these steps. The template files in the `RunnerTemplate` folder will help you get started quickly with vendor-specific implementations.

Contributions, especially PRs implementing **Metal** and **HIP** support, are highly encouraged!

### 1. Create a New Vendor Directory

- First, create a directory for the new vendor under `cross_gpgpu/`. For example, if the new vendor is "MyVendor", create the directory as follows:

   ```bash
   mkdir cross_gpgpu/MyVendor
   ```

### 2. Copy RunnerTemplate

- Copy the template files from the `RunnerTemplate` folder to the new vendor directory.

   ```bash
   cp -r cross_gpgpu/RunnerTemplate/* cross_gpgpu/MyVendor/
   ```

### 3. Modify `TemplateImpl.cpp`

- `TemplateImpl.cpp` is the core implementation file for the new vendor. Below is a key portion of this file:

   ```cpp
   #include "RunnerInterface.hpp"
   //include your gpgpu kernel codes.

   // Genv: Structure to hold the GPGPU environment settings and resources.
   struct Genv{
       //
   };

   // Gcodes: Structure to manage and store GPGPU kernel codes.
   struct Gcodes{
       //
   };

   // InitEnv: Initializes the GPGPU environment and kernel code structures.
   // Allocates memory for 'env' (Genv) and 'kens' (Gcodes).
   void
   Runner::InitEnv()
   {
       env = new Genv;
       kens = new Gcodes;
   }
   
   void
   Runner::UnInit()
   {

   }
   // BuildKernel: Compiles or prepares the GPGPU kernel for execution.
   void
   Runner::BuildKernel()
   {
       //
   }

   // ActivateSTFT: Executes the Short-Time Fourier Transform (STFT) on the input data using GPGPU.
   MAYBE_DATA
   Runner::ActivateSTFT(   VECF& inData, 
                           const int& windowRadix, 
                           const float& overlapRatio)
   {
       // Default code blocks
       const unsigned int  FullSize    = inData.size();
       const int           windowSize  = 1 << windowRadix;
       const int           qtConst     = toQuot(FullSize, overlapRatio, windowSize);
       const unsigned int  OFullSize   = qtConst * windowSize;
       const unsigned int  OHalfSize   = OFullSize / 2;
       const unsigned int  OMove       = windowSize * (1.0f - overlapRatio);
       // End default

       // Memory allocation samples
       auto tempMem = new complex  [OFullSize]();
       auto qtBuffer= new float    [qtConst]();
       std::vector<float> outMem(OHalfSize);

       // Implement your GPGPU kernel execution code here.

       delete[] tempMem;
       delete[] qtBuffer;

       return std::move(outMem);
   }
   ```

   Modify the `Genv` and `Gcodes` structures to suit the new vendor, and implement the `BuildKernel` and `ActivateSTFT` functions.

### 4. CMake Configuration

- Modify a `CMakeLists.txt` file in the new vendor directory to define the build settings. For example:

### 5. Build and Test

- Build the new vendor and ensure the executable is generated. Test the executable to verify it works as expected.



---

By adding a new vendor, this project can support a broader range of environments for STFT processing. If you need further assistance, feel free to reach out.
