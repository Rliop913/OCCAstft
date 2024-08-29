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
      // printf("OIDX: %u, Origin: %u, WIDX: %u, WLocIdx: %u\n",
      // overlapIdx, originIdx, windowIdx, windowLocalIdx);
      outReal[overlapIdx] = inData[originIdx * exceeded];
    }
  }
}


// @kernel void DCRemove_Common(
//     float* outReal,
//     const unsigned int OFullSize,
//     const unsigned int windowRadix
// )
// {
//     for(unsigned int o_itr=0; o_itr < OFullSize; o_itr += 1024; @outer)
//     {
//         @shared float windowBuffer[8192];//for removing DC
//         for(unsigned int i_itr=0; i_itr < 1024; ++i_itr; @inner)
//         {
//             const unsigned int overlapIdx = o_itr + i_itr;
//             const unsigned int windowIdx = (overlapIdx >> windowRadix);
//             const unsigned int windowLocalIdx = overlapIdx & ((1 << windowIdx) - 1);
//             outReal[overlapIdx] = inData[originIdx];
//         }
//     }
// }

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
      float LeftReal = inReal[OIdx];
      float LeftImag = inImag[OIdx];
      float RightReal = inReal[OIdx + HwindowSize];
      float RightImag = inImag[OIdx + HwindowSize];
      unsigned int segmentSize = 1 << stageRadix;
      unsigned int segmentItr = OIdx >> stageRadix;
      unsigned int segmentIndex = OIdx & (segmentSize - 1);
      unsigned int LeftStoreIdx = segmentItr * (segmentSize << 1) + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + segmentSize;
      // printf("L: %u, R: %u\n", LeftStoreIdx, RightStoreIdx);
      complex tw = twiddle(
        segmentK(OIdx, segmentSize, HwindowSize),
        HwindowSize << 1
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

