#define _USE_MATH_DEFINES
#include <math.h>
#include <math.h>
#include <cstdio>

typedef struct complex_t {
  float real, imag;
} complex;

typedef struct pairs_t {
  unsigned int first, second;
} pairs;

__device__ inline float window_func(const int index,
                                    const int window_size) {
  float normalized_index = (float) index;
  normalized_index /= ((float) (window_size - 1));
  float angle = 2.0f * M_PI * normalized_index;
  return 0.5f * (1.0f - cos(angle));
}

__device__ inline int reverseBits(int num,
                                  int radix_2_data) {
  int reversed = 0;
  for (int i = 0; i < radix_2_data; ++i) {
    reversed = (reversed << 1) | (num & 1);
    num >>= 1;
  }
  return reversed;
}

__device__ pairs indexer(const unsigned int firstMaximumID,
                         const int powed_stage) {
  pairs temp;
  temp.first = firstMaximumID + (firstMaximumID & (~(powed_stage - 1)));
  temp.second = temp.first + powed_stage;
  return temp;
}

__device__ inline int segmentK(const int lsave,
                               const int segmentSize,
                               const int HwindowSize) {
  return ((lsave % segmentSize) * HwindowSize) / segmentSize;
}

__device__ complex twiddle(int k,
                           int windowSize) {
  complex temp;
  float angle = -2.0 * M_PI * ((float) k / (float) windowSize);
  temp.real = cos(angle);
  temp.imag = sin(angle);
  return temp;
}

__device__ inline complex cmult(const complex a,
                                const complex b) {
  complex result;
  result.real = a.real * b.real - a.imag * b.imag;
  result.imag = a.real * b.imag + a.imag * b.real;
  return result;
}

__device__ inline complex cadd(complex a,
                               const complex b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

__device__ inline complex csub(complex a,
                               const complex b) {
  a.real -= b.real;
  a.imag -= b.imag;
  return a;
}

__device__ inline float cmod(complex a) {
  return (sqrt(
    a.real * a.real + a.imag * a.imag
  ));
}

__device__ inline void DaCAdd(const int i_itr,
                              const unsigned int Half,
                              float windowAdded[]) {
  unsigned int inRange = i_itr < Half;
  float Dpoint = windowAdded[i_itr];
  float Apoint = windowAdded[i_itr + (Half * inRange)];
  windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
}

extern "C" __global__ __launch_bounds__(256) void _occa_overlap_N_window_0(float * in,
                                                                           complex * buffer,
                                                                           const unsigned int fullSize,
                                                                           const unsigned int OFullSize,
                                                                           const int windowSize,
                                                                           const unsigned int OMove) {
  {
    unsigned int w_num = 0 + (256 * blockIdx.x);
    {
      int w_itr = 0 + threadIdx.x;
      unsigned int FID = w_num + w_itr;
      unsigned int read_point = (unsigned int) ((FID) / windowSize) * OMove + ((FID) % windowSize);

      // printf("%u buffer overlap\n", OMove);//buffer[FID].imag);
      buffer[FID].real = read_point >= fullSize ? 0.0 : in[read_point];
      // * window_func((FID) % windowSize, windowSize);
      buffer[FID].imag = 0.0;
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void _occa_bitReverse_temp_0(complex * buffer,
                                                                          complex * result,
                                                                          const unsigned int OFullSize,
                                                                          const int windowSize,
                                                                          const int radixData) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int w_itr = 0 + threadIdx.x;
      unsigned int Gidx = (o_itr + w_itr);
      unsigned int Lidx = (Gidx % windowSize);
      unsigned int dst_idx = reverseBits(Lidx, radixData);
      unsigned int BID = Gidx - Lidx + dst_idx;
      result[BID] = buffer[Gidx];
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void _occa_Butterfly_0(complex * buffer,
                                                                    const int windowSize,
                                                                    const int powed_stage,
                                                                    const unsigned int OHalfSize,
                                                                    const int radixData) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int GID = o_itr + i_itr;
      pairs butterfly_target = indexer(GID, powed_stage);
      int k = (GID % powed_stage) * (windowSize / (2 * powed_stage));
      //(GID%powed_stage) * (windowSize / powed_stage);
      complex this_twiddle = twiddle(k, windowSize);
      complex first = buffer[butterfly_target.first];
      complex second = buffer[butterfly_target.second];
      complex tempx = cadd(first, second);
      complex tempy = csub(first, second);
      tempy = cmult(tempy, this_twiddle);
      buffer[butterfly_target.first] = tempx;
      buffer[butterfly_target.second] = tempy;
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void _occa_toPower_0(complex * buffer,
                                                                  float * out,
                                                                  const unsigned int OHalfSize,
                                                                  const int windowRadix) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int i_itr = 0 + threadIdx.x;
      const unsigned int GID = o_itr + i_itr;
      unsigned int BID = (GID >> (windowRadix - 1)) * (1 << windowRadix) + (GID & ((1 << (windowRadix - 1)) - 1));
      float powered = cmod(buffer[BID]);
      //powered = log10(powered);
      out[GID] = powered;
    }
  }
}

extern "C" __global__ __launch_bounds__(512) void _occa_preprocessed_ODW10_STH_STFT_0(float * inData,
                                                                                      const unsigned int qtConst,
                                                                                      const unsigned int fullSize,
                                                                                      const unsigned int OMove,
                                                                                      const unsigned int OHalfSize,
                                                                                      complex * bufferOut) {
  {
    unsigned int o_itr = 0 + (512 * blockIdx.x);
    __shared__ complex FBank[1024];
    __shared__ complex SBank[1024];
    __shared__ float windowAdded[512];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int q_itr = o_itr >> 9;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 512;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 512].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 512].imag = 0;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (512)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 1024.0);
      FBank[i_itr].real *= window_func(i_itr, 1024);
      FBank[i_itr + 512].real -= (windowAdded[0] / 1024.0);
      FBank[i_itr + 512].real *= window_func(i_itr + 512, 1024);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 512), 1024);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 512), 1024);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 512), 1024);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 512), 1024);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 512), 1024);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 512), 1024);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 512), 1024);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 512), 1024);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 512), 1024);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 512), 1024);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 512] = storeR;
    }
  }
}

extern "C" __global__ __launch_bounds__(512) void _occa_preprocesses_ODW_10_0(float * inData,
                                                                              const unsigned int qtConst,
                                                                              const unsigned int fullSize,
                                                                              const unsigned int OMove,
                                                                              complex * bufferOut) {
  {
    unsigned int o_itr = 0 + blockIdx.x;
    __shared__ complex windowBuffer[1024];
    __shared__ float windowAdded[512];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 512;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 512].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 512].imag = 0;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 512;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (512 * inRange)].real;
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 1024.0);
      windowBuffer[i_itr + 512].real -= (windowAdded[0] / 1024.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 1024);
      windowBuffer[i_itr + 512].real *= window_func(i_itr + 512, 1024);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      bufferOut[o_itr * 1024 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 1024 + i_itr + 512] = windowBuffer[i_itr + 512];
    }
  }
}

extern "C" __global__ __launch_bounds__(512) void _occa_Stockhpotimized10_0(complex * buffer,
                                                                            const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (512 * blockIdx.x);
    __shared__ complex FBank[1024];
    __shared__ complex SBank[1024];
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 512), 1024);
      complex LEFT = buffer[o_itr * 2 + i_itr];
      complex RIGHT = buffer[o_itr * 2 + i_itr + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 512), 1024);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 512), 1024);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 512), 1024);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 512), 1024);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 512), 1024);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 512), 1024);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 512), 1024);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 512), 1024);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 512), 1024);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 512] = storeR;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocesses_ODW_11_0(float * inData,
                                                                               const unsigned int qtConst,
                                                                               const unsigned int fullSize,
                                                                               const unsigned int OMove,
                                                                               complex * bufferOut) {
  {
    unsigned int o_itr = 0 + blockIdx.x;
    __shared__ complex windowBuffer[2048];
    __shared__ float windowAdded[1024];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 1024;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 1024].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 1024].imag = 0;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (1024)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 2048.0);
      windowBuffer[i_itr + 1024].real -= (windowAdded[0] / 2048.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 2048);
      windowBuffer[i_itr + 1024].real *= window_func(i_itr + 1024, 2048);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      bufferOut[o_itr * 2048 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 2048 + i_itr + 1024] = windowBuffer[i_itr + 1024];
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_Stockhpotimized11_0(complex * buffer,
                                                                             const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (1024 * blockIdx.x);
    __shared__ complex FBank[2048];
    __shared__ complex SBank[2048];
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 1024), 2048);
      complex LEFT = buffer[o_itr * 2 + i_itr];
      complex RIGHT = buffer[o_itr * 2 + i_itr + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 1024), 2048);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 1024] = storeR;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocessed_ODW11_STH_STFT_0(float * inData,
                                                                                       const unsigned int qtConst,
                                                                                       const unsigned int fullSize,
                                                                                       const unsigned int OMove,
                                                                                       const unsigned int OHalfSize,
                                                                                       complex * bufferOut) {
  {
    unsigned int o_itr = 0 + (1024 * blockIdx.x);
    __shared__ complex FBank[2048];
    __shared__ complex SBank[2048];
    __shared__ float windowAdded[1024];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int q_itr = o_itr >> 10;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 1024;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 1024].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 1024].imag = 0;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (1024)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 2048.0);
      FBank[i_itr].real *= window_func(i_itr, 2048);
      FBank[i_itr + 1024].real -= (windowAdded[0] / 2048.0);
      FBank[i_itr + 1024].real *= window_func(i_itr + 1024, 2048);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 1024), 2048);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 1024), 2048);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 1024] = storeR;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocesses_ODW_12_0(float * inData,
                                                                               const unsigned int qtConst,
                                                                               const unsigned int fullSize,
                                                                               const unsigned int OMove,
                                                                               complex * bufferOut) {
  {
    unsigned int o_itr = 0 + blockIdx.x;
    __shared__ complex windowBuffer[4096];
    __shared__ float windowAdded[2048];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 2048;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 2048].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 2048].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 2048;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 2048].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 2048].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (2048)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (2048)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 4096.0);
      windowBuffer[i_itr + 2048].real -= (windowAdded[0] / 4096.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 4096);
      windowBuffer[i_itr + 2048].real *= window_func(i_itr + 2048, 4096);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 4096.0);
      windowBuffer[i_itr + 2048].real -= (windowAdded[0] / 4096.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 4096);
      windowBuffer[i_itr + 2048].real *= window_func(i_itr + 2048, 4096);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      bufferOut[o_itr * 4096 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 4096 + i_itr + 2048] = windowBuffer[i_itr + 2048];
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      bufferOut[o_itr * 4096 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 4096 + i_itr + 2048] = windowBuffer[i_itr + 2048];
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_Stockhpotimized12_0(complex * buffer,
                                                                             const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (2048 * blockIdx.x);
    __shared__ complex FBank[4096];
    __shared__ complex SBank[4096];
    {
      int i_itr = 0 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 2048), 4096);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 2048];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 2048] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 2048), 4096);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 2048];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 2048] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 2048), 4096);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 2048), 4096);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 2048] = storeR;
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocessed_ODW12_STH_STFT_0(float * inData,
                                                                                       const unsigned int qtConst,
                                                                                       const unsigned int fullSize,
                                                                                       const unsigned int OMove,
                                                                                       const unsigned int OHalfSize,
                                                                                       complex * bufferOut) {
  {
    unsigned int o_itr = 0 + (2048 * blockIdx.x);
    __shared__ complex FBank[4096];
    __shared__ complex SBank[4096];
    __shared__ float windowAdded[2048];
    {
      int i_itr = 0 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 11;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 2048;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 2048].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 2048].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 11;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 2048;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 2048].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 2048].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (2048)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (2048)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 4096.0);
      FBank[i_itr].real *= window_func(i_itr, 4096);
      FBank[i_itr + 2048].real -= (windowAdded[0] / 4096.0);
      FBank[i_itr + 2048].real *= window_func(i_itr + 2048, 4096);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 4096.0);
      FBank[i_itr].real *= window_func(i_itr, 4096);
      FBank[i_itr + 2048].real -= (windowAdded[0] / 4096.0);
      FBank[i_itr + 2048].real *= window_func(i_itr + 2048, 4096);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 2048), 4096);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 2048), 4096);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 2048), 4096);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 2048), 4096);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 2048] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 2048), 4096);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 2048] = storeR;
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocesses_ODW_13_0(float * inData,
                                                                               const unsigned int qtConst,
                                                                               const unsigned int fullSize,
                                                                               const unsigned int OMove,
                                                                               complex * bufferOut) {
  {
    unsigned int o_itr = 0 + blockIdx.x;
    __shared__ complex windowBuffer[8192];
    __shared__ float windowAdded[4096];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 4096;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 4096].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 4096].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 4096;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 4096].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 4096].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 4096;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 4096].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 4096].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 4096;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 4096].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 4096].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 8192);
      windowBuffer[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 8192);
      windowBuffer[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 8192);
      windowBuffer[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 8192);
      windowBuffer[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      bufferOut[o_itr * 8192 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 8192 + i_itr + 4096] = windowBuffer[i_itr + 4096];
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      bufferOut[o_itr * 8192 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 8192 + i_itr + 4096] = windowBuffer[i_itr + 4096];
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      bufferOut[o_itr * 8192 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 8192 + i_itr + 4096] = windowBuffer[i_itr + 4096];
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      bufferOut[o_itr * 8192 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 8192 + i_itr + 4096] = windowBuffer[i_itr + 4096];
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_Stockhpotimized13_0(complex * buffer,
                                                                             const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (4096 * blockIdx.x);
    __shared__ complex FBank[8192];
    __shared__ complex SBank[8192];
    {
      int i_itr = 0 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 4096];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 4096] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 4096];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 4096] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 4096];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 4096] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 4096];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 4096] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 4096), 8192);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 4096), 8192);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 4096), 8192);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 4096), 8192);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 4096] = storeR;
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocessed_ODW13_STH_STFT_0(float * inData,
                                                                                       const unsigned int qtConst,
                                                                                       const unsigned int fullSize,
                                                                                       const unsigned int OMove,
                                                                                       const unsigned int OHalfSize,
                                                                                       complex * bufferOut) {
  {
    unsigned int o_itr = 0 + (4096 * blockIdx.x);
    __shared__ complex FBank[8192];
    __shared__ complex SBank[8192];
    __shared__ float windowAdded[4096];
    {
      int i_itr = 0 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 12;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 4096;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 4096].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 4096].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 12;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 4096;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 4096].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 4096].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 12;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 4096;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 4096].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 4096].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 12;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 4096;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 4096].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 4096].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr].real *= window_func(i_itr, 8192);
      FBank[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr].real *= window_func(i_itr, 8192);
      FBank[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr].real *= window_func(i_itr, 8192);
      FBank[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr].real *= window_func(i_itr, 8192);
      FBank[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 4096), 8192);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 4096), 8192);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 4096), 8192);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 4096), 8192);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 4096] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 4096), 8192);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 4096] = storeR;
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocesses_ODW_14_0(float * inData,
                                                                               const unsigned int qtConst,
                                                                               const unsigned int fullSize,
                                                                               const unsigned int OMove,
                                                                               complex * bufferOut) {
  {
    unsigned int o_itr = 0 + blockIdx.x;
    __shared__ complex windowBuffer[16384];
    __shared__ float windowAdded[8192];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_Stockhpotimized14_0(complex * buffer,
                                                                             const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (8192 * blockIdx.x);
    __shared__ complex FBank[16384];
    __shared__ complex SBank[16384];
    {
      int i_itr = 0 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 8192];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 8192] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 8192];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 8192] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 8192];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 8192] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 8192];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 8192] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 8192];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 8192] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 8192];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 8192] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 8192];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 8192] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 8192];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        FBank[i_itr] = storeL;
        FBank[i_itr + 8192] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocessed_ODW14_STH_STFT_0(float * inData,
                                                                                       const unsigned int qtConst,
                                                                                       const unsigned int fullSize,
                                                                                       const unsigned int OMove,
                                                                                       const unsigned int OHalfSize,
                                                                                       complex * bufferOut) {
  {
    unsigned int o_itr = 0 + (8192 * blockIdx.x);
    __shared__ complex FBank[16384];
    __shared__ complex SBank[16384];
    __shared__ float windowAdded[8192];
    {
      int i_itr = 0 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 13;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 13;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 13;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 13;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 13;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 13;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 13;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 13;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 8192;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 8192].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 8192].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 8192), 16384);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 8192), 16384);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 8192] = storeR;
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocesses_ODW_15_0(float * inData,
                                                                               const unsigned int qtConst,
                                                                               const unsigned int fullSize,
                                                                               const unsigned int OMove,
                                                                               complex * bufferOut) {
  {
    unsigned int o_itr = 0 + blockIdx.x;
    __shared__ complex windowBuffer[32768];
    __shared__ float windowAdded[16384];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      windowBuffer[i_itr].real = inData[idx] * isOverflowed;
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      windowBuffer[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_Stockhpotimized15_0(complex * buffer,
                                                                             const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (16384 * blockIdx.x);
    __shared__ complex FBank[32768];
    __shared__ complex SBank[32768];
    {
      int i_itr = 0 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      {
        complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
        complex LEFT = buffer[o_itr * 2 + i_itr];
        complex RIGHT = buffer[o_itr * 2 + i_itr + 16384];
        complex storeL = cadd(LEFT, RIGHT);
        complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
        SBank[i_itr] = storeL;
        SBank[i_itr + 16384] = storeR;
      }
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 12) << 13);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 11) << 12);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      buffer[o_itr * 2 + i_itr] = storeL;
      buffer[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocessed_ODW15_STH_STFT_0(float * inData,
                                                                                       const unsigned int qtConst,
                                                                                       const unsigned int fullSize,
                                                                                       const unsigned int OMove,
                                                                                       const unsigned int OHalfSize,
                                                                                       complex * bufferOut) {
  {
    unsigned int o_itr = 0 + (16384 * blockIdx.x);
    __shared__ complex FBank[32768];
    __shared__ complex SBank[32768];
    __shared__ float windowAdded[16384];
    {
      int i_itr = 0 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      //overlap and extends
      unsigned int q_itr = o_itr >> 14;
      unsigned int idx = q_itr * OMove + i_itr;
      unsigned int Ridx = q_itr * OMove + i_itr + 16384;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      FBank[i_itr].real = inData[idx] * isOverflowed;
      FBank[i_itr].imag = 0;
      FBank[i_itr + 16384].real = inData[Ridx] * RisOverflowed;
      FBank[i_itr + 16384].imag = 0;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 8191) | ((i_itr >> 13) << 14);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 4095) | ((i_itr >> 12) << 13);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 2047) | ((i_itr >> 11) << 12);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1023) | ((i_itr >> 10) << 11);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 16384), 32768);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 1024 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 2048 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 3072 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 4096 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 5120 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 6144 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 7168 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 8192 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 9216 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 10240 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 11264 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 12288 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 13312 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 14336 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
    __syncthreads();
    {
      int i_itr = 15360 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 16384), 32768);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      bufferOut[o_itr * 2 + i_itr] = storeL;
      bufferOut[o_itr * 2 + i_itr + 16384] = storeR;
      ;
    }
  }
}

