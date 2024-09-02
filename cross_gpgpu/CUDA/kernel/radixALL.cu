#define _USE_MATH_DEFINES
#include <math.h>

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
  return (Ra * Ib) + (Ia * Rb);
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


// inline
// void
// DaCAdd( const int i_itr,
//         const unsigned int Half,
//         float windowAdded[])
// {
// }

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

extern "C" __global__ __launch_bounds__(256) void _occa_toPower_0(float * out,
                                                                  float * Real,
                                                                  float * Imag,
                                                                  const unsigned int OFullSize,
                                                                  const int windowRadix) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int i_itr = 0 + threadIdx.x;
      const unsigned int GID = o_itr + i_itr;
      float R = Real[GID];
      float I = Imag[GID];
      out[GID] = sqrt(R * R + I * I);
    }
  }
}

extern "C" __global__ __launch_bounds__(512) void _occa_preprocessed_ODW10_STH_STFT_0(float * inData,
                                                                                      const unsigned int qtConst,
                                                                                      const unsigned int fullSize,
                                                                                      const unsigned int OMove,
                                                                                      const unsigned int OHalfSize,
                                                                                      float * Rout,
                                                                                      float * Iout) {
  {
    unsigned int o_itr = 0 + (512 * blockIdx.x);
    __shared__ float FRBank[1024];
    __shared__ float FIBank[1024];
    __shared__ float SRBank[1024];
    __shared__ float SIBank[1024];
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
      FRBank[i_itr] = inData[idx] * isOverflowed;
      FIBank[i_itr] = 0;
      FRBank[i_itr + 512] = inData[Ridx] * RisOverflowed;
      FIBank[i_itr + 512] = 0;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = FRBank[i_itr];
      float Apoint = FRBank[i_itr + (512)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 256;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (256 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 128;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (128 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 64;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (64 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 32;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (32 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 16;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (16 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 8;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (8 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 4;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (4 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 2;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (2 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 1;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (1 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      FRBank[i_itr] -= (windowAdded[0] / 1024.0);
      FRBank[i_itr] *= window_func(i_itr, 1024);
      FRBank[i_itr + 512] -= (windowAdded[0] / 1024.0);
      FRBank[i_itr + 512] *= window_func(i_itr + 512, 1024);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 512), 1024);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[i_itr];
      LEFT.imag = 0.0;
      RIGHT.real = FRBank[i_itr + 512];
      RIGHT.imag = 0.0;
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 512), 1024);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 256];
      RIGHT.imag = SIBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 512] = storeR.real;
      FIBank[i_itr + 512] = storeR.imag;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 512), 1024);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 128];
      RIGHT.imag = FIBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 512), 1024);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 64];
      RIGHT.imag = SIBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 512] = storeR.real;
      FIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 512), 1024);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 32];
      RIGHT.imag = FIBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 512), 1024);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 16];
      RIGHT.imag = SIBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 512] = storeR.real;
      FIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 512), 1024);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 8];
      RIGHT.imag = FIBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 512), 1024);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 4];
      RIGHT.imag = SIBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 512] = storeR.real;
      FIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 512), 1024);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 2];
      RIGHT.imag = FIBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 512), 1024);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 1];
      RIGHT.imag = SIBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      Rout[o_itr * 2 + i_itr] = storeL.real;
      Iout[o_itr * 2 + i_itr] = storeL.imag;
      Rout[o_itr * 2 + i_itr + 512] = storeR.real;
      Iout[o_itr * 2 + i_itr + 512] = storeR.imag;
    }
  }
}

extern "C" __global__ __launch_bounds__(512) void _occa_preprocesses_ODW_10_0(float * inData,
                                                                              const unsigned int qtConst,
                                                                              const unsigned int fullSize,
                                                                              const unsigned int OMove,
                                                                              float * Rout) {
  {
    unsigned int o_itr = 0 + blockIdx.x;
    __shared__ float wr[1024];
    __shared__ float windowAdded[512];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 512;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      wr[i_itr] = inData[idx] * isOverflowed;
      wr[i_itr + 512] = inData[Ridx] * RisOverflowed;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 512;
      float Dpoint = wr[i_itr];
      float Apoint = wr[i_itr + (512 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 256;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (256 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 128;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (128 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 64;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (64 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 32;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (32 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 16;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (16 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 8;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (8 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 4;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (4 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 2;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (2 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 1;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (1 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      wr[i_itr] -= (windowAdded[0] / 1024.0);
      wr[i_itr + 512] -= (windowAdded[0] / 1024.0);
      wr[i_itr] *= window_func(i_itr, 1024);
      wr[i_itr + 512] *= window_func(i_itr + 512, 1024);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      Rout[o_itr * 1024 + i_itr] = wr[i_itr];
      Rout[o_itr * 1024 + i_itr + 512] = wr[i_itr + 512];
    }
  }
}

extern "C" __global__ __launch_bounds__(512) void _occa_Stockhpotimized10_0(float * Rout,
                                                                            float * Iout,
                                                                            const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (512 * blockIdx.x);
    __shared__ float FRBank[1024];
    __shared__ float FIBank[1024];
    __shared__ float SRBank[1024];
    __shared__ float SIBank[1024];
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 512), 1024);
      complex LEFT;
      complex RIGHT;
      LEFT.real = Rout[o_itr * 2 + i_itr];
      LEFT.imag = Iout[o_itr * 2 + i_itr];
      RIGHT.real = Rout[o_itr * 2 + i_itr + 512];
      RIGHT.imag = Iout[o_itr * 2 + i_itr + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 512), 1024);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 256];
      RIGHT.imag = SIBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 512] = storeR.real;
      FIBank[i_itr + 512] = storeR.imag;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 512), 1024);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 128];
      RIGHT.imag = FIBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 512), 1024);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 64];
      RIGHT.imag = SIBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 512] = storeR.real;
      FIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 512), 1024);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 32];
      RIGHT.imag = FIBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 512), 1024);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 16];
      RIGHT.imag = SIBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 512] = storeR.real;
      FIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 512), 1024);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 8];
      RIGHT.imag = FIBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 512), 1024);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 4];
      RIGHT.imag = SIBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 512] = storeR.real;
      FIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 512), 1024);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 2];
      RIGHT.imag = FIBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 512] = storeR.real;
      SIBank[i_itr + 512] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 512), 1024);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 1];
      RIGHT.imag = SIBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      Rout[o_itr * 2 + i_itr] = storeL.real;
      Iout[o_itr * 2 + i_itr] = storeL.imag;
      Rout[o_itr * 2 + i_itr + 512] = storeR.real;
      Iout[o_itr * 2 + i_itr + 512] = storeR.imag;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocesses_ODW_11_0(float * inData,
                                                                               const unsigned int qtConst,
                                                                               const unsigned int fullSize,
                                                                               const unsigned int OMove,
                                                                               float * Rout) {
  {
    unsigned int o_itr = 0 + blockIdx.x;
    __shared__ float wr[2048];
    __shared__ float windowAdded[1024];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 1024;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      wr[i_itr] = inData[idx] * isOverflowed;
      wr[i_itr] = 0;
      wr[i_itr + 1024] = inData[Ridx] * RisOverflowed;
      wr[i_itr + 1024] = 0;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = wr[i_itr];
      float Apoint = wr[i_itr + (1024)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 512;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (512 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 256;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (256 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 128;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (128 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 64;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (64 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 32;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (32 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 16;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (16 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 8;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (8 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 4;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (4 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 2;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (2 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 1;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (1 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      wr[i_itr] -= (windowAdded[0] / 2048.0);
      wr[i_itr + 1024] -= (windowAdded[0] / 2048.0);
      wr[i_itr] *= window_func(i_itr, 2048);
      wr[i_itr + 1024] *= window_func(i_itr + 1024, 2048);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      Rout[o_itr * 2048 + i_itr] = wr[i_itr];
      Rout[o_itr * 2048 + i_itr + 1024] = wr[i_itr + 1024];
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_Stockhpotimized11_0(float * Rout,
                                                                             float * Iout,
                                                                             const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (1024 * blockIdx.x);
    __shared__ float FRBank[2048];
    __shared__ float FIBank[2048];
    __shared__ float SRBank[2048];
    __shared__ float SIBank[2048];
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 1024), 2048);
      complex LEFT;
      complex RIGHT;
      LEFT.real = Rout[o_itr * 2 + i_itr];
      LEFT.imag = Iout[o_itr * 2 + i_itr];
      RIGHT.real = Rout[o_itr * 2 + i_itr + 1024];
      RIGHT.imag = Iout[o_itr * 2 + i_itr + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 512];
      RIGHT.imag = SIBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 256];
      RIGHT.imag = FIBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 128];
      RIGHT.imag = SIBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 64];
      RIGHT.imag = FIBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 32];
      RIGHT.imag = SIBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 16];
      RIGHT.imag = FIBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 8];
      RIGHT.imag = SIBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 4];
      RIGHT.imag = FIBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 2];
      RIGHT.imag = SIBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 1024), 2048);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 1];
      RIGHT.imag = FIBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      Rout[o_itr * 2 + i_itr] = storeL.real;
      Iout[o_itr * 2 + i_itr] = storeL.imag;
      Rout[o_itr * 2 + i_itr + 1024] = storeR.real;
      Iout[o_itr * 2 + i_itr + 1024] = storeR.imag;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_preprocessed_ODW11_STH_STFT_0(float * inData,
                                                                                       const unsigned int qtConst,
                                                                                       const unsigned int fullSize,
                                                                                       const unsigned int OMove,
                                                                                       const unsigned int OHalfSize,
                                                                                       float * Rout,
                                                                                       float * Iout) {
  {
    unsigned int o_itr = 0 + (1024 * blockIdx.x);
    __shared__ float FRBank[2048];
    __shared__ float FIBank[2048];
    __shared__ float SRBank[2048];
    __shared__ float SIBank[2048];
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
      FRBank[i_itr] = inData[idx] * isOverflowed;
      FIBank[i_itr] = 0;
      FRBank[i_itr + 1024] = inData[Ridx] * RisOverflowed;
      FIBank[i_itr + 1024] = 0;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      float Dpoint = FRBank[i_itr];
      float Apoint = FRBank[i_itr + (1024)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 512;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (512 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 256;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (256 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 128;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (128 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 64;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (64 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 32;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (32 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 16;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (16 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 8;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (8 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 4;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (4 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 2;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (2 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int inRange = i_itr < 1;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (1 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      FRBank[i_itr] -= (windowAdded[0] / 2048.0);
      FRBank[i_itr] *= window_func(i_itr, 2048);
      FRBank[i_itr + 1024] -= (windowAdded[0] / 2048.0);
      FRBank[i_itr + 1024] *= window_func(i_itr + 1024, 2048);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 1024), 2048);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[i_itr];
      LEFT.imag = FIBank[i_itr];
      RIGHT.real = FRBank[i_itr + 1024];
      RIGHT.imag = FIBank[i_itr + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 511) | ((i_itr >> 9) << 10);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 512];
      RIGHT.imag = SIBank[LeftIndex + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
      ;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 256];
      RIGHT.imag = FIBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 128];
      RIGHT.imag = SIBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 64];
      RIGHT.imag = FIBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 32];
      RIGHT.imag = SIBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 16];
      RIGHT.imag = FIBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 8];
      RIGHT.imag = SIBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 4];
      RIGHT.imag = FIBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SRBank[i_itr] = storeL.real;
      SIBank[i_itr] = storeL.imag;
      SRBank[i_itr + 1024] = storeR.real;
      SIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT;
      complex RIGHT;
      LEFT.real = SRBank[LeftIndex];
      LEFT.imag = SIBank[LeftIndex];
      RIGHT.real = SRBank[LeftIndex + 2];
      RIGHT.imag = SIBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FRBank[i_itr] = storeL.real;
      FIBank[i_itr] = storeL.imag;
      FRBank[i_itr + 1024] = storeR.real;
      FIBank[i_itr + 1024] = storeR.imag;
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      complex thisTwiddle = twiddle(segmentK(i_itr, 1, 1024), 2048);
      unsigned int LeftIndex = (i_itr << 1);
      complex LEFT;
      complex RIGHT;
      LEFT.real = FRBank[LeftIndex];
      LEFT.imag = FIBank[LeftIndex];
      RIGHT.real = FRBank[LeftIndex + 1];
      RIGHT.imag = FIBank[LeftIndex + 1];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      Rout[o_itr * 2 + i_itr] = storeL.real;
      Iout[o_itr * 2 + i_itr] = storeL.imag;
      Rout[o_itr * 2 + i_itr + 1024] = storeR.real;
      Iout[o_itr * 2 + i_itr + 1024] = storeR.imag;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_Overlap_Common_0(float * inData,
                                                                          const unsigned int OFullSize,
                                                                          const unsigned int fullSize,
                                                                          const unsigned int windowRadix,
                                                                          const unsigned int OMove,
                                                                          float * outReal) {
  {
    unsigned int o_itr = 0 + (1024 * blockIdx.x);
    {
      unsigned int i_itr = 0 + threadIdx.x;
      const unsigned int overlapIdx = o_itr + i_itr;
      const unsigned int windowIdx = (overlapIdx >> windowRadix);
      const unsigned int windowLocalIdx = overlapIdx & ((1 << windowRadix) - 1);
      const unsigned int originIdx = windowIdx * OMove + windowLocalIdx;
      const unsigned int exceeded = originIdx < fullSize;
      outReal[overlapIdx] = inData[originIdx * exceeded] * exceeded;
    }
  }
}

extern "C" __global__ __launch_bounds__(1024) void _occa_Window_Common_0(float * outReal,
                                                                         const unsigned int OFullSize,
                                                                         const unsigned int windowRadix) {
  {
    unsigned int o_itr = 0 + (1024 * blockIdx.x);
    {
      unsigned int i_itr = 0 + threadIdx.x;
      unsigned int Gidx = o_itr + i_itr;
      outReal[Gidx] *= window_func((Gidx & (windowRadix - 1)), 1 << windowRadix);
    }
  }
}

extern "C" __global__ __launch_bounds__(64) void _occa_DCRemove_Common_0(float * outReal,
                                                                         const unsigned int OFullSize,
                                                                         const unsigned int windowSize) {
  {
    unsigned int o_itr = 0 + (windowSize * blockIdx.x);
    __shared__ float added[128];
    //for removing DC
    {
      unsigned int inititr = 0 + threadIdx.x;
      added[inititr] = 0;
    }
    __syncthreads();
    for (unsigned int windowItr = 0; windowItr < windowSize; windowItr += 64) {
      {
        unsigned int i_itr = 0 + threadIdx.x;
        added[i_itr + 64] = outReal[o_itr + windowItr + i_itr];
      }
      __syncthreads();
      {
        unsigned int i_itr = 0 + threadIdx.x;
        added[i_itr] += added[i_itr + 64];
      }
      __syncthreads();
    }
    for (unsigned int segment = 32; segment > 0; segment >>= 1) {
      {
        unsigned int i_itr = 0 + threadIdx.x;
        unsigned int inSegment = i_itr < segment;
        float left = added[i_itr];
        float right = added[i_itr + segment];
        added[i_itr] = (left + right) * inSegment;
      }
      __syncthreads();
    }
    for (unsigned int windowItr = 0; windowItr < windowSize; windowItr += 64) {
      {
        unsigned int i_itr = 0 + threadIdx.x;
        outReal[o_itr + windowItr + i_itr] -= (added[0] / (float) windowSize);
      }
      __syncthreads();
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void _occa_StockHamDITCommon_0(float * inReal,
                                                                            float * inImag,
                                                                            float * outReal,
                                                                            float * outImag,
                                                                            const unsigned int HwindowSize,
                                                                            const unsigned int stageRadix,
                                                                            const unsigned int OHalfSize,
                                                                            const unsigned int radixData) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      unsigned int i_itr = 0 + threadIdx.x;
      unsigned int OIdx = o_itr + i_itr;
      unsigned int FwindowSize = HwindowSize << 1;
      unsigned int GlobalItr = OIdx >> (radixData - 1);
      unsigned int GlobalIndex = (OIdx & (HwindowSize - 1));
      OIdx = GlobalItr * FwindowSize + GlobalIndex;
      float LeftReal = inReal[OIdx];
      float LeftImag = inImag[OIdx];
      float RightReal = inReal[OIdx + HwindowSize];
      float RightImag = inImag[OIdx + HwindowSize];
      unsigned int segmentSize = 1 << stageRadix;
      unsigned int segmentItr = GlobalIndex >> stageRadix;
      unsigned int segmentIndex = (GlobalIndex & (segmentSize - 1));
      //OIdx & (segmentSize - 1);
      unsigned int LeftStoreIdx = segmentItr * (segmentSize << 1) + segmentIndex + GlobalItr * FwindowSize;
      unsigned int RightStoreIdx = LeftStoreIdx + segmentSize;
      complex tw = twiddle(
        segmentK(OIdx, segmentSize, HwindowSize),
        FwindowSize
      );
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      outReal[LeftStoreIdx] = LeftReal + RTwid;
      outImag[LeftStoreIdx] = LeftImag + ITwid;
      outReal[RightStoreIdx] = LeftReal - RTwid;
      outImag[RightStoreIdx] = LeftImag - ITwid;
    }
  }
}

