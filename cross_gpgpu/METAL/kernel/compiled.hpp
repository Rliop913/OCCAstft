#define _USE_MATH_DEFINES
#include <metal_compute>

#include <metal_stdlib>

using namespace metal;
#include <math.h>
// #include <math.h>

typedef struct complex_t {
  float real, imag;
} complex;

typedef struct pairs_t {
  unsigned int first, second;
} pairs;

inline float window_func(const int index,
                         const int window_size) {
  float normalized_index = (float) index;
  normalized_index /= ((float) (window_size - 1));
  float angle = 2.0f * M_PI * normalized_index;
  return 0.5f * (1.0f - cos(angle));
}

inline int reverseBits(int num,
                       int radix_2_data) {
  int reversed = 0;
  for (int i = 0; i < radix_2_data; ++i) {
    reversed = (reversed << 1) | (num & 1);
    num >>= 1;
  }
  return reversed;
}

pairs indexer(const unsigned int firstMaximumID,
              const int powed_stage) {
  pairs temp;
  temp.first = firstMaximumID + (firstMaximumID & (~(powed_stage - 1)));
  temp.second = temp.first + powed_stage;
  return temp;
}

inline int segmentK(const int lsave,
                    const int segmentSize,
                    const int HwindowSize) {
  return ((lsave % segmentSize) * HwindowSize) / segmentSize;
}

complex twiddle(int k,
                int windowSize) {
  complex temp;
  float angle = -2.0 * M_PI * ((float) k / (float) windowSize);
  temp.real = cos(angle);
  temp.imag = sin(angle);
  return temp;
}

inline complex cmult(const complex a,
                     const complex b) {
  complex result;
  result.real = a.real * b.real - a.imag * b.imag;
  result.imag = a.real * b.imag + a.imag * b.real;
  return result;
}

inline float RMult(const float Ra,
                   const float Rb,
                   const float Ia,
                   const float Ib) {
  return (Ra * Rb) - (Ia * Ib);
}

inline float IMult(const float Ra,
                   const float Rb,
                   const float Ia,
                   const float Ib) {
  return (Ra * Ib) + (Ia * Rb);
}

inline complex cadd(complex a,
                    const complex b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

inline complex csub(complex a,
                    const complex b) {
  a.real -= b.real;
  a.imag -= b.imag;
  return a;
}

inline float cmod(complex a) {
  return (sqrt(
    a.real * a.real + a.imag * a.imag
  ));
}

kernel void _occa_bitReverse_temp_0(device complex * buffer [[buffer(0)]],
                                    device complex * result [[buffer(1)]],
                                    constant unsigned int & OFullSize [[buffer(2)]],
                                    constant int & windowSize [[buffer(3)]],
                                    constant int & radixData [[buffer(4)]],
                                    uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                    uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  {
    unsigned int o_itr = 0 + (256 * _occa_group_position.x);
    {
      int w_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = (o_itr + w_itr);
      unsigned int Lidx = (Gidx % windowSize);
      unsigned int dst_idx = reverseBits(Lidx, radixData);
      unsigned int BID = Gidx - Lidx + dst_idx;
      result[BID] = buffer[Gidx];
    }
  }
}

kernel void _occa_toPower_0(device float * out [[buffer(0)]],
                            device float * Real [[buffer(1)]],
                            device float * Imag [[buffer(2)]],
                            constant unsigned int & OFullSize [[buffer(3)]],
                            constant int & windowRadix [[buffer(4)]],
                            uint3 _occa_group_position [[threadgroup_position_in_grid]],
                            uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  {
    unsigned int o_itr = 0 + (256 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
      const unsigned int GID = o_itr + i_itr;
      float R = Real[GID];
      float I = Imag[GID];
      out[GID] = sqrt(R * R + I * I);
    }
  }
}

kernel void _occa_Stockhpotimized6_0(device float * Rout [[buffer(0)]],
                                     device float * Iout [[buffer(1)]],
                                     constant unsigned int & OHalfSize [[buffer(2)]],
                                     uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                     uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float SIBank[64];
  threadgroup float SRBank[64];
  threadgroup float FIBank[64];
  threadgroup float FRBank[64];
  {
    unsigned int o_itr = 0 + (32 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 5;
      unsigned int GlobalIndex = (Gidx & (32 - 1));
      Gidx = GlobalItr * 64 + GlobalIndex;
      float LeftReal = Rout[Gidx];
      float LeftImag = 0;
      float RightReal = Rout[Gidx + 32];
      float RightImag = 0;
      unsigned int LeftStoreIdx = i_itr * 2;
      unsigned int RightStoreIdx = LeftStoreIdx + 1;
      complex tw\
 = twiddle(segmentK(i_itr, 1, 32), 64);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 32];
      float RightImag = FIBank[i_itr + 32];
      unsigned int segmentItr = i_itr >> 1;
      unsigned int segmentIndex = (i_itr & (2 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 4 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 2;
      complex tw\
 = twiddle(segmentK(i_itr, 2, 32), 64);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 32];
      float RightImag = SIBank[i_itr + 32];
      unsigned int segmentItr = i_itr >> 2;
      unsigned int segmentIndex = (i_itr & (4 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 8 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 4;
      complex tw\
 = twiddle(segmentK(i_itr, 4, 32), 64);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 32];
      float RightImag = FIBank[i_itr + 32];
      unsigned int segmentItr = i_itr >> 3;
      unsigned int segmentIndex = (i_itr & (8 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 16 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 8;
      complex tw\
 = twiddle(segmentK(i_itr, 8, 32), 64);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 32];
      float RightImag = SIBank[i_itr + 32];
      unsigned int segmentItr = i_itr >> 4;
      unsigned int segmentIndex = (i_itr & (16 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 32 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 16;
      complex tw\
 = twiddle(segmentK(i_itr, 16, 32), 64);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 5;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 32];
      float RightImag = FIBank[i_itr + 32];
      unsigned int LeftStoreIdx\
 = i_itr + GlobalItr * 64;
      unsigned int RightStoreIdx = LeftStoreIdx + 32;
      complex tw\
 = twiddle(segmentK(i_itr, 32, 32), 64);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      Rout[LeftStoreIdx] = LeftReal + RTwid;
      Iout[LeftStoreIdx] = LeftImag + ITwid;
      Rout[RightStoreIdx] = LeftReal - RTwid;
      Iout[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
  }
}

kernel void _occa_Stockhpotimized7_0(device float * Rout [[buffer(0)]],
                                     device float * Iout [[buffer(1)]],
                                     constant unsigned int & OHalfSize [[buffer(2)]],
                                     uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                     uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float SIBank[128];
  threadgroup float SRBank[128];
  threadgroup float FIBank[128];
  threadgroup float FRBank[128];
  {
    unsigned int o_itr = 0 + (64 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 6;
      unsigned int GlobalIndex = (Gidx & (64 - 1));
      Gidx = GlobalItr * 128 + GlobalIndex;
      float LeftReal = Rout[Gidx];
      float LeftImag = 0;
      float RightReal = Rout[Gidx + 64];
      float RightImag = 0;
      unsigned int LeftStoreIdx = i_itr * 2;
      unsigned int RightStoreIdx = LeftStoreIdx + 1;
      complex tw\
 = twiddle(segmentK(i_itr, 1, 64), 128);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 64];
      float RightImag = FIBank[i_itr + 64];
      unsigned int segmentItr = i_itr >> 1;
      unsigned int segmentIndex = (i_itr & (2 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 4 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 2;
      complex tw\
 = twiddle(segmentK(i_itr, 2, 64), 128);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 64];
      float RightImag = SIBank[i_itr + 64];
      unsigned int segmentItr = i_itr >> 2;
      unsigned int segmentIndex = (i_itr & (4 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 8 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 4;
      complex tw\
 = twiddle(segmentK(i_itr, 4, 64), 128);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 64];
      float RightImag = FIBank[i_itr + 64];
      unsigned int segmentItr = i_itr >> 3;
      unsigned int segmentIndex = (i_itr & (8 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 16 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 8;
      complex tw\
 = twiddle(segmentK(i_itr, 8, 64), 128);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 64];
      float RightImag = SIBank[i_itr + 64];
      unsigned int segmentItr = i_itr >> 4;
      unsigned int segmentIndex = (i_itr & (16 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 32 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 16;
      complex tw\
 = twiddle(segmentK(i_itr, 16, 64), 128);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 64];
      float RightImag = FIBank[i_itr + 64];
      unsigned int segmentItr = i_itr >> 5;
      unsigned int segmentIndex = (i_itr & (32 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 64 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 32;
      complex tw\
 = twiddle(segmentK(i_itr, 32, 64), 128);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 6;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 64];
      float RightImag = SIBank[i_itr + 64];
      unsigned int LeftStoreIdx\
 = i_itr + GlobalItr * 128;
      unsigned int RightStoreIdx = LeftStoreIdx + 64;
      complex tw\
 = twiddle(segmentK(i_itr, 64, 64), 128);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      Rout[LeftStoreIdx] = LeftReal + RTwid;
      Iout[LeftStoreIdx] = LeftImag + ITwid;
      Rout[RightStoreIdx] = LeftReal - RTwid;
      Iout[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
  }
}

kernel void _occa_Stockhpotimized8_0(device float * Rout [[buffer(0)]],
                                     device float * Iout [[buffer(1)]],
                                     constant unsigned int & OHalfSize [[buffer(2)]],
                                     uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                     uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float SIBank[256];
  threadgroup float SRBank[256];
  threadgroup float FIBank[256];
  threadgroup float FRBank[256];
  {
    unsigned int o_itr = 0 + (128 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 7;
      unsigned int GlobalIndex = (Gidx & (128 - 1));
      Gidx = GlobalItr * 256 + GlobalIndex;
      float LeftReal = Rout[Gidx];
      float LeftImag = 0;
      float RightReal = Rout[Gidx + 128];
      float RightImag = 0;
      unsigned int LeftStoreIdx = i_itr * 2;
      unsigned int RightStoreIdx = LeftStoreIdx + 1;
      complex tw\
 = twiddle(segmentK(i_itr, 1, 128), 256);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 128];
      float RightImag = FIBank[i_itr + 128];
      unsigned int segmentItr = i_itr >> 1;
      unsigned int segmentIndex = (i_itr & (2 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 4 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 2;
      complex tw\
 = twiddle(segmentK(i_itr, 2, 128), 256);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 128];
      float RightImag = SIBank[i_itr + 128];
      unsigned int segmentItr = i_itr >> 2;
      unsigned int segmentIndex = (i_itr & (4 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 8 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 4;
      complex tw\
 = twiddle(segmentK(i_itr, 4, 128), 256);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 128];
      float RightImag = FIBank[i_itr + 128];
      unsigned int segmentItr = i_itr >> 3;
      unsigned int segmentIndex = (i_itr & (8 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 16 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 8;
      complex tw\
 = twiddle(segmentK(i_itr, 8, 128), 256);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 128];
      float RightImag = SIBank[i_itr + 128];
      unsigned int segmentItr = i_itr >> 4;
      unsigned int segmentIndex = (i_itr & (16 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 32 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 16;
      complex tw\
 = twiddle(segmentK(i_itr, 16, 128), 256);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 128];
      float RightImag = FIBank[i_itr + 128];
      unsigned int segmentItr = i_itr >> 5;
      unsigned int segmentIndex = (i_itr & (32 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 64 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 32;
      complex tw\
 = twiddle(segmentK(i_itr, 32, 128), 256);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 128];
      float RightImag = SIBank[i_itr + 128];
      unsigned int segmentItr = i_itr >> 6;
      unsigned int segmentIndex = (i_itr & (64 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 128 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 64;
      complex tw\
 = twiddle(segmentK(i_itr, 64, 128), 256);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 7;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 128];
      float RightImag = FIBank[i_itr + 128];
      unsigned int LeftStoreIdx\
 = i_itr + GlobalItr * 256;
      unsigned int RightStoreIdx = LeftStoreIdx + 128;
      complex tw\
 = twiddle(segmentK(i_itr, 128, 128), 256);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      Rout[LeftStoreIdx] = LeftReal + RTwid;
      Iout[LeftStoreIdx] = LeftImag + ITwid;
      Rout[RightStoreIdx] = LeftReal - RTwid;
      Iout[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
  }
}

kernel void _occa_Stockhpotimized9_0(device float * Rout [[buffer(0)]],
                                     device float * Iout [[buffer(1)]],
                                     constant unsigned int & OHalfSize [[buffer(2)]],
                                     uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                     uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float SIBank[512];
  threadgroup float SRBank[512];
  threadgroup float FIBank[512];
  threadgroup float FRBank[512];
  {
    unsigned int o_itr = 0 + (256 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 8;
      unsigned int GlobalIndex = (Gidx & (256 - 1));
      Gidx = GlobalItr * 512 + GlobalIndex;
      float LeftReal = Rout[Gidx];
      float LeftImag = 0;
      float RightReal = Rout[Gidx + 256];
      float RightImag = 0;
      unsigned int LeftStoreIdx = i_itr * 2;
      unsigned int RightStoreIdx = LeftStoreIdx + 1;
      complex tw\
 = twiddle(segmentK(i_itr, 1, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 256];
      float RightImag = FIBank[i_itr + 256];
      unsigned int segmentItr = i_itr >> 1;
      unsigned int segmentIndex = (i_itr & (2 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 4 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 2;
      complex tw\
 = twiddle(segmentK(i_itr, 2, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 256];
      float RightImag = SIBank[i_itr + 256];
      unsigned int segmentItr = i_itr >> 2;
      unsigned int segmentIndex = (i_itr & (4 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 8 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 4;
      complex tw\
 = twiddle(segmentK(i_itr, 4, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 256];
      float RightImag = FIBank[i_itr + 256];
      unsigned int segmentItr = i_itr >> 3;
      unsigned int segmentIndex = (i_itr & (8 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 16 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 8;
      complex tw\
 = twiddle(segmentK(i_itr, 8, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 256];
      float RightImag = SIBank[i_itr + 256];
      unsigned int segmentItr = i_itr >> 4;
      unsigned int segmentIndex = (i_itr & (16 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 32 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 16;
      complex tw\
 = twiddle(segmentK(i_itr, 16, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 256];
      float RightImag = FIBank[i_itr + 256];
      unsigned int segmentItr = i_itr >> 5;
      unsigned int segmentIndex = (i_itr & (32 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 64 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 32;
      complex tw\
 = twiddle(segmentK(i_itr, 32, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 256];
      float RightImag = SIBank[i_itr + 256];
      unsigned int segmentItr = i_itr >> 6;
      unsigned int segmentIndex = (i_itr & (64 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 128 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 64;
      complex tw\
 = twiddle(segmentK(i_itr, 64, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 256];
      float RightImag = FIBank[i_itr + 256];
      unsigned int segmentItr = i_itr >> 7;
      unsigned int segmentIndex = (i_itr & (128 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 256 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 128;
      complex tw\
 = twiddle(segmentK(i_itr, 128, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 8;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 256];
      float RightImag = SIBank[i_itr + 256];
      unsigned int LeftStoreIdx\
 = i_itr + GlobalItr * 512;
      unsigned int RightStoreIdx = LeftStoreIdx + 256;
      complex tw\
 = twiddle(segmentK(i_itr, 256, 256), 512);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      Rout[LeftStoreIdx] = LeftReal + RTwid;
      Iout[LeftStoreIdx] = LeftImag + ITwid;
      Rout[RightStoreIdx] = LeftReal - RTwid;
      Iout[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
  }
}

kernel void _occa_preprocessed_ODW10_STH_STFT_0(device float * inData [[buffer(0)]],
                                                constant unsigned int & qtConst [[buffer(1)]],
                                                constant unsigned int & fullSize [[buffer(2)]],
                                                constant unsigned int & OMove [[buffer(3)]],
                                                constant unsigned int & OHalfSize [[buffer(4)]],
                                                device float * Rout [[buffer(5)]],
                                                device float * Iout [[buffer(6)]],
                                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float windowAdded[512];
  threadgroup float SIBank[1024];
  threadgroup float SRBank[1024];
  threadgroup float FIBank[1024];
  threadgroup float FRBank[1024];
  {
    unsigned int o_itr = 0 + (512 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float Dpoint = FRBank[i_itr];
      float Apoint = FRBank[i_itr + (512)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 256;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (256 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 128;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (128 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 64;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (64 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 32;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (32 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 16;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (16 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 8;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (8 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 4;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (4 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 2;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (2 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 1;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (1 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      FRBank[i_itr] -= (windowAdded[0] / 1024.0);
      FRBank[i_itr] *= window_func(i_itr, 1024);
      FRBank[i_itr + 512] -= (windowAdded[0] / 1024.0);
      FRBank[i_itr + 512] *= window_func(i_itr + 512, 1024);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 0;
      unsigned int segmentIndex = (i_itr & (1 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 2 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 1;
      complex tw\
 = twiddle(segmentK(i_itr, 1, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 1;
      unsigned int segmentIndex = (i_itr & (2 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 4 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 2;
      complex tw\
 = twiddle(segmentK(i_itr, 2, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 2;
      unsigned int segmentIndex = (i_itr & (4 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 8 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 4;
      complex tw\
 = twiddle(segmentK(i_itr, 4, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 3;
      unsigned int segmentIndex = (i_itr & (8 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 16 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 8;
      complex tw\
 = twiddle(segmentK(i_itr, 8, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 4;
      unsigned int segmentIndex = (i_itr & (16 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 32 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 16;
      complex tw\
 = twiddle(segmentK(i_itr, 16, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 5;
      unsigned int segmentIndex = (i_itr & (32 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 64 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 32;
      complex tw\
 = twiddle(segmentK(i_itr, 32, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 6;
      unsigned int segmentIndex = (i_itr & (64 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 128 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 64;
      complex tw\
 = twiddle(segmentK(i_itr, 64, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 7;
      unsigned int segmentIndex = (i_itr & (128 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 256 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 128;
      complex tw\
 = twiddle(segmentK(i_itr, 128, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 8;
      unsigned int segmentIndex = (i_itr & (256 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 512 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 256;
      complex tw\
 = twiddle(segmentK(i_itr, 256, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 9;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int LeftStoreIdx\
 = i_itr + GlobalItr * 1024;
      unsigned int RightStoreIdx = LeftStoreIdx + 512;
      complex tw\
 = twiddle(segmentK(i_itr, 512, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      Rout[LeftStoreIdx] = LeftReal + RTwid;
      Iout[LeftStoreIdx] = LeftImag + ITwid;
      Rout[RightStoreIdx] = LeftReal - RTwid;
      Iout[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
  }
}

kernel void _occa_preprocesses_ODW_10_0(device float * inData [[buffer(0)]],
                                        constant unsigned int & qtConst [[buffer(1)]],
                                        constant unsigned int & fullSize [[buffer(2)]],
                                        constant unsigned int & OMove [[buffer(3)]],
                                        device float * Rout [[buffer(4)]],
                                        uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                        uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float windowAdded[512];
  threadgroup float wr[1024];
  {
    unsigned int o_itr = 0 + _occa_group_position.x;
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 512;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      wr[i_itr] = inData[idx] * isOverflowed;
      wr[i_itr + 512] = inData[Ridx] * RisOverflowed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 512;
      float Dpoint = wr[i_itr];
      float Apoint = wr[i_itr + (512 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 256;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (256 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 128;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (128 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 64;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (64 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 32;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (32 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 16;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (16 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 8;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (8 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 4;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (4 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 2;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (2 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 1;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (1 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      wr[i_itr] -= (windowAdded[0] / 1024.0);
      wr[i_itr + 512] -= (windowAdded[0] / 1024.0);
      wr[i_itr] *= window_func(i_itr, 1024);
      wr[i_itr + 512] *= window_func(i_itr + 512, 1024);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      Rout[o_itr * 1024 + i_itr] = wr[i_itr];
      Rout[o_itr * 1024 + i_itr + 512] = wr[i_itr + 512];
    }
  }
}

kernel void _occa_Stockhpotimized10_0(device float * Rout [[buffer(0)]],
                                      device float * Iout [[buffer(1)]],
                                      constant unsigned int & OHalfSize [[buffer(2)]],
                                      uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                      uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float SIBank[1024];
  threadgroup float SRBank[1024];
  threadgroup float FIBank[1024];
  threadgroup float FRBank[1024];
  {
    unsigned int o_itr = 0 + (512 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 9;
      unsigned int GlobalIndex = (Gidx & (512 - 1));
      Gidx = GlobalItr * 1024 + GlobalIndex;
      float LeftReal = Rout[Gidx];
      float LeftImag = 0;
      float RightReal = Rout[Gidx + 512];
      float RightImag = 0;
      unsigned int LeftStoreIdx = i_itr * 2;
      unsigned int RightStoreIdx = LeftStoreIdx + 1;
      complex tw\
 = twiddle(segmentK(i_itr, 1, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 1;
      unsigned int segmentIndex = (i_itr & (2 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 4 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 2;
      complex tw\
 = twiddle(segmentK(i_itr, 2, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 2;
      unsigned int segmentIndex = (i_itr & (4 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 8 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 4;
      complex tw\
 = twiddle(segmentK(i_itr, 4, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 3;
      unsigned int segmentIndex = (i_itr & (8 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 16 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 8;
      complex tw\
 = twiddle(segmentK(i_itr, 8, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 4;
      unsigned int segmentIndex = (i_itr & (16 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 32 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 16;
      complex tw\
 = twiddle(segmentK(i_itr, 16, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 5;
      unsigned int segmentIndex = (i_itr & (32 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 64 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 32;
      complex tw\
 = twiddle(segmentK(i_itr, 32, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 6;
      unsigned int segmentIndex = (i_itr & (64 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 128 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 64;
      complex tw\
 = twiddle(segmentK(i_itr, 64, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 7;
      unsigned int segmentIndex = (i_itr & (128 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 256 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 128;
      complex tw\
 = twiddle(segmentK(i_itr, 128, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 512];
      float RightImag = SIBank[i_itr + 512];
      unsigned int segmentItr = i_itr >> 8;
      unsigned int segmentIndex = (i_itr & (256 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 512 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 256;
      complex tw\
 = twiddle(segmentK(i_itr, 256, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 9;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 512];
      float RightImag = FIBank[i_itr + 512];
      unsigned int LeftStoreIdx\
 = i_itr + GlobalItr * 1024;
      unsigned int RightStoreIdx = LeftStoreIdx + 512;
      complex tw\
 = twiddle(segmentK(i_itr, 512, 512), 1024);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      Rout[LeftStoreIdx] = LeftReal + RTwid;
      Iout[LeftStoreIdx] = LeftImag + ITwid;
      Rout[RightStoreIdx] = LeftReal - RTwid;
      Iout[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
  }
}


// @kernel void Stockhpotimized10(
//     float* Rout,
//     float* Iout,
//     const unsigned int OHalfSize)
// {
//     for(unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 512; @outer)
//     {
//         @shared float FRBank[1024];
//         @shared float FIBank[1024];
//         @shared float SRBank[1024];
//         @shared float SIBank[1024];
//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             complex thisTwiddle = twiddle(segmentK(i_itr, 512, 512), 1024);
//             complex LEFT;
//             complex RIGHT;
//             LEFT.real = Rout[o_itr * 2 + i_itr];
//             LEFT.imag = Iout[o_itr * 2 + i_itr];
//             RIGHT.real= Rout[o_itr * 2 + i_itr + 512];
//             RIGHT.imag= Iout[o_itr * 2 + i_itr + 512];
//             complex storeL = cadd(LEFT, RIGHT);
//             complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
//             SRBank[i_itr] = storeL.real;
//             SIBank[i_itr] = storeL.imag;
//             SRBank[i_itr + 512] = storeR.real;
//             SIBank[i_itr + 512] = storeR.imag;
//         }

//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             OpSthmSecondTF(256, 255, 8, 9, 512, 1024);
//         }

//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             OpSthmFirstTS(128, 127, 7, 8, 512, 1024)
//         }

//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             OpSthmSecondTF(64, 63, 6, 7, 512, 1024)
//         }
//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             OpSthmFirstTS(32, 31, 5, 6, 512, 1024)
//         }
//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             OpSthmSecondTF(16, 15, 4, 5, 512, 1024)
//         }
//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             OpSthmFirstTS(8, 7, 3, 4, 512, 1024)
//         }
//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             OpSthmSecondTF(4, 3, 2, 3, 512, 1024)
//         }
//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             OpSthmFirstTS(2, 1, 1, 2, 512, 1024)
//         }
//         for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
//         {
//             complex thisTwiddle = twiddle(segmentK(i_itr, 1, 512), 1024);
//             unsigned int LeftIndex =  (i_itr << 1);
//             complex LEFT;
//             complex RIGHT;
//             LEFT.real = SRBank[LeftIndex];
//             LEFT.imag = SIBank[LeftIndex];
//             RIGHT.real= SRBank[LeftIndex + 1];
//             RIGHT.imag= SIBank[LeftIndex + 1];
//             complex storeL = cadd(LEFT, RIGHT);
//             complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);

//             Rout[o_itr * 2 + i_itr] = storeL.real;
//             Iout[o_itr * 2 + i_itr] = storeL.imag;
//             Rout[o_itr * 2 + i_itr + 512] = storeR.real;
//             Iout[o_itr * 2 + i_itr + 512] = storeR.imag;
//         }
//     }
// }

kernel void _occa_preprocesses_ODW_11_0(device float * inData [[buffer(0)]],
                                        constant unsigned int & qtConst [[buffer(1)]],
                                        constant unsigned int & fullSize [[buffer(2)]],
                                        constant unsigned int & OMove [[buffer(3)]],
                                        device float * Rout [[buffer(4)]],
                                        uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                        uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float windowAdded[1024];
  threadgroup float wr[2048];
  {
    unsigned int o_itr = 0 + _occa_group_position.x;
    {
      int i_itr = 0 + _occa_thread_position.x;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float Dpoint = wr[i_itr];
      float Apoint = wr[i_itr + (1024)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 512;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (512 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 256;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (256 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 128;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (128 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 64;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (64 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 32;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (32 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 16;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (16 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 8;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (8 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 4;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (4 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 2;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (2 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 1;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (1 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      wr[i_itr] -= (windowAdded[0] / 2048.0);
      wr[i_itr + 1024] -= (windowAdded[0] / 2048.0);
      wr[i_itr] *= window_func(i_itr, 2048);
      wr[i_itr + 1024] *= window_func(i_itr + 1024, 2048);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      Rout[o_itr * 2048 + i_itr] = wr[i_itr];
      Rout[o_itr * 2048 + i_itr + 1024] = wr[i_itr + 1024];
    }
  }
}

kernel void _occa_Stockhpotimized11_0(device float * Rout [[buffer(0)]],
                                      device float * Iout [[buffer(1)]],
                                      constant unsigned int & OHalfSize [[buffer(2)]],
                                      uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                      uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float SIBank[2048];
  threadgroup float SRBank[2048];
  threadgroup float FIBank[2048];
  threadgroup float FRBank[2048];
  {
    unsigned int o_itr = 0 + (1024 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 10;
      unsigned int GlobalIndex = (Gidx & (1024 - 1));
      Gidx = GlobalItr * 2048 + GlobalIndex;
      float LeftReal = Rout[Gidx];
      float LeftImag = 0;
      float RightReal = Rout[Gidx + 1024];
      float RightImag = 0;
      unsigned int LeftStoreIdx = i_itr * 2;
      unsigned int RightStoreIdx = LeftStoreIdx + 1;
      complex tw\
 = twiddle(segmentK(i_itr, 1, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 1;
      unsigned int segmentIndex = (i_itr & (2 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 4 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 2;
      complex tw\
 = twiddle(segmentK(i_itr, 2, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 2;
      unsigned int segmentIndex = (i_itr & (4 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 8 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 4;
      complex tw\
 = twiddle(segmentK(i_itr, 4, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 3;
      unsigned int segmentIndex = (i_itr & (8 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 16 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 8;
      complex tw\
 = twiddle(segmentK(i_itr, 8, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 4;
      unsigned int segmentIndex = (i_itr & (16 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 32 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 16;
      complex tw\
 = twiddle(segmentK(i_itr, 16, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 5;
      unsigned int segmentIndex = (i_itr & (32 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 64 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 32;
      complex tw\
 = twiddle(segmentK(i_itr, 32, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 6;
      unsigned int segmentIndex = (i_itr & (64 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 128 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 64;
      complex tw\
 = twiddle(segmentK(i_itr, 64, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 7;
      unsigned int segmentIndex = (i_itr & (128 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 256 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 128;
      complex tw\
 = twiddle(segmentK(i_itr, 128, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 8;
      unsigned int segmentIndex = (i_itr & (256 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 512 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 256;
      complex tw\
 = twiddle(segmentK(i_itr, 256, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 9;
      unsigned int segmentIndex = (i_itr & (512 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 1024 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 512;
      complex tw\
 = twiddle(segmentK(i_itr, 512, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 10;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int LeftStoreIdx\
 = i_itr + GlobalItr * 2048;
      unsigned int RightStoreIdx = LeftStoreIdx + 1024;
      complex tw\
 = twiddle(segmentK(i_itr, 1024, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      Rout[LeftStoreIdx] = LeftReal + RTwid;
      Iout[LeftStoreIdx] = LeftImag + ITwid;
      Rout[RightStoreIdx] = LeftReal - RTwid;
      Iout[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
  }
}

kernel void _occa_preprocessed_ODW11_STH_STFT_0(device float * inData [[buffer(0)]],
                                                constant unsigned int & qtConst [[buffer(1)]],
                                                constant unsigned int & fullSize [[buffer(2)]],
                                                constant unsigned int & OMove [[buffer(3)]],
                                                constant unsigned int & OHalfSize [[buffer(4)]],
                                                device float * Rout [[buffer(5)]],
                                                device float * Iout [[buffer(6)]],
                                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float windowAdded[1024];
  threadgroup float SIBank[2048];
  threadgroup float SRBank[2048];
  threadgroup float FIBank[2048];
  threadgroup float FRBank[2048];
  {
    unsigned int o_itr = 0 + (1024 * _occa_group_position.x);
    {
      int i_itr = 0 + _occa_thread_position.x;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float Dpoint = FRBank[i_itr];
      float Apoint = FRBank[i_itr + (1024)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 512;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (512 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 256;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (256 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 128;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (128 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 64;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (64 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 32;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (32 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 16;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (16 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 8;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (8 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 4;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (4 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 2;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (2 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int inRange = i_itr < 1;
      float Dpoint = windowAdded[i_itr];
      float Apoint = windowAdded[i_itr + (1 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      FRBank[i_itr] -= (windowAdded[0] / 2048.0);
      FRBank[i_itr] *= window_func(i_itr, 2048);
      FRBank[i_itr + 1024] -= (windowAdded[0] / 2048.0);
      FRBank[i_itr + 1024] *= window_func(i_itr + 1024, 2048);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 0;
      unsigned int segmentIndex = (i_itr & (1 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 2 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 1;
      complex tw\
 = twiddle(segmentK(i_itr, 1, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 1;
      unsigned int segmentIndex = (i_itr & (2 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 4 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 2;
      complex tw\
 = twiddle(segmentK(i_itr, 2, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 2;
      unsigned int segmentIndex = (i_itr & (4 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 8 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 4;
      complex tw\
 = twiddle(segmentK(i_itr, 4, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 3;
      unsigned int segmentIndex = (i_itr & (8 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 16 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 8;
      complex tw\
 = twiddle(segmentK(i_itr, 8, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 4;
      unsigned int segmentIndex = (i_itr & (16 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 32 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 16;
      complex tw\
 = twiddle(segmentK(i_itr, 16, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 5;
      unsigned int segmentIndex = (i_itr & (32 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 64 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 32;
      complex tw\
 = twiddle(segmentK(i_itr, 32, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 6;
      unsigned int segmentIndex = (i_itr & (64 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 128 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 64;
      complex tw\
 = twiddle(segmentK(i_itr, 64, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 7;
      unsigned int segmentIndex = (i_itr & (128 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 256 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 128;
      complex tw\
 = twiddle(segmentK(i_itr, 128, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 8;
      unsigned int segmentIndex = (i_itr & (256 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 512 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 256;
      complex tw\
 = twiddle(segmentK(i_itr, 256, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      SRBank[LeftStoreIdx] = LeftReal + RTwid;
      SIBank[LeftStoreIdx] = LeftImag + ITwid;
      SRBank[RightStoreIdx] = LeftReal - RTwid;
      SIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      float LeftReal = SRBank[i_itr];
      float LeftImag = SIBank[i_itr];
      float RightReal = SRBank[i_itr + 1024];
      float RightImag = SIBank[i_itr + 1024];
      unsigned int segmentItr = i_itr >> 9;
      unsigned int segmentIndex = (i_itr & (512 - 1));
      unsigned int LeftStoreIdx\
 = segmentItr * 1024 + segmentIndex;
      unsigned int RightStoreIdx = LeftStoreIdx + 512;
      complex tw\
 = twiddle(segmentK(i_itr, 512, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      FRBank[LeftStoreIdx] = LeftReal + RTwid;
      FIBank[LeftStoreIdx] = LeftImag + ITwid;
      FRBank[RightStoreIdx] = LeftReal - RTwid;
      FIBank[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      unsigned int GlobalItr = Gidx >> 10;
      float LeftReal = FRBank[i_itr];
      float LeftImag = FIBank[i_itr];
      float RightReal = FRBank[i_itr + 1024];
      float RightImag = FIBank[i_itr + 1024];
      unsigned int LeftStoreIdx\
 = i_itr + GlobalItr * 2048;
      unsigned int RightStoreIdx = LeftStoreIdx + 1024;
      complex tw\
 = twiddle(segmentK(i_itr, 1024, 1024), 2048);
      float RTwid = RMult(RightReal, tw.real, RightImag, tw.imag);
      float ITwid = IMult(RightReal, tw.real, RightImag, tw.imag);
      Rout[LeftStoreIdx] = LeftReal + RTwid;
      Iout[LeftStoreIdx] = LeftImag + ITwid;
      Rout[RightStoreIdx] = LeftReal - RTwid;
      Iout[RightStoreIdx] = LeftImag - ITwid;
      ;
    }
  }
}

kernel void _occa_Overlap_Common_0(device float * inData [[buffer(0)]],
                                   constant unsigned int & OFullSize [[buffer(1)]],
                                   constant unsigned int & fullSize [[buffer(2)]],
                                   constant unsigned int & windowRadix [[buffer(3)]],
                                   constant unsigned int & OMove [[buffer(4)]],
                                   device float * outReal [[buffer(5)]],
                                   uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                   uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  {
    unsigned int o_itr = 0 + (64 * _occa_group_position.x);
    {
      unsigned int i_itr = 0 + _occa_thread_position.x;
      const unsigned int overlapIdx = o_itr + i_itr;
      const unsigned int windowIdx = (overlapIdx >> windowRadix);
      const unsigned int windowLocalIdx = overlapIdx & ((1 << windowRadix) - 1);
      const unsigned int originIdx = windowIdx * OMove + windowLocalIdx;
      const unsigned int exceeded = originIdx < fullSize;
      outReal[overlapIdx] = inData[originIdx * exceeded] * exceeded;
    }
  }
}

kernel void _occa_Window_Common_0(device float * outReal [[buffer(0)]],
                                  constant unsigned int & OFullSize [[buffer(1)]],
                                  constant unsigned int & windowRadix [[buffer(2)]],
                                  uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                  uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  {
    unsigned int o_itr = 0 + (64 * _occa_group_position.x);
    {
      unsigned int i_itr = 0 + _occa_thread_position.x;
      unsigned int Gidx = o_itr + i_itr;
      outReal[Gidx] *= window_func((Gidx & (windowRadix - 1)), 1 << windowRadix);
    }
  }
}

kernel void _occa_DCRemove_Common_0(device float * outReal [[buffer(0)]],
                                    constant unsigned int & OFullSize [[buffer(1)]],
                                    constant unsigned int & windowSize [[buffer(2)]],
                                    uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                    uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  threadgroup float added[128];
  {
    unsigned int o_itr = 0 + (windowSize * _occa_group_position.x);
    //for removing DC
    {
      unsigned int inititr = 0 + _occa_thread_position.x;
      added[inititr] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (unsigned int windowItr = 0; windowItr < windowSize; windowItr += 64) {
      {
        unsigned int i_itr = 0 + _occa_thread_position.x;
        added[i_itr + 64] = outReal[o_itr + windowItr + i_itr];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      {
        unsigned int i_itr = 0 + _occa_thread_position.x;
        added[i_itr] += added[i_itr + 64];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (unsigned int segment = 32; segment > 0; segment >>= 1) {
      {
        unsigned int i_itr = 0 + _occa_thread_position.x;
        unsigned int inSegment = i_itr < segment;
        float left = added[i_itr];
        float right = added[i_itr + segment];
        added[i_itr] = (left + right) * inSegment;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (unsigned int windowItr = 0; windowItr < windowSize; windowItr += 64) {
      {
        unsigned int i_itr = 0 + _occa_thread_position.x;
        outReal[o_itr + windowItr + i_itr] -= (added[0] / (float) windowSize);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

kernel void _occa_StockHamDITCommon_0(device float * inReal [[buffer(0)]],
                                      device float * inImag [[buffer(1)]],
                                      device float * outReal [[buffer(2)]],
                                      device float * outImag [[buffer(3)]],
                                      constant unsigned int & HwindowSize [[buffer(4)]],
                                      constant unsigned int & stageRadix [[buffer(5)]],
                                      constant unsigned int & OHalfSize [[buffer(6)]],
                                      constant unsigned int & radixData [[buffer(7)]],
                                      uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                      uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
  {
    unsigned int o_itr = 0 + (256 * _occa_group_position.x);
    {
      unsigned int i_itr = 0 + _occa_thread_position.x;
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
