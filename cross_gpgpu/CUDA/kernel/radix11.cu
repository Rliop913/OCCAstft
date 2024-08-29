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

__device__ inline float RMult(const float Ra,
                              const float Rb,
                              const float Ia,
                              const float Ib) {
  return (Ra * Rb) - (Ia * Ib);
}

__device__ inline float IMult(const float Ra,
                              const float Rb,
                              const float Ia,
                              const float Ib) {
  return (Ra * Rb) + (Ia * Ib);
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
                              complex windowAdded[]) {
  unsigned int inRange = i_itr < Half;
  float Dpoint = windowAdded[i_itr].imag;
  float Apoint = windowAdded[i_itr + (Half * inRange)].imag;
  windowAdded[i_itr].imag = (Dpoint + Apoint) * inRange;
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

