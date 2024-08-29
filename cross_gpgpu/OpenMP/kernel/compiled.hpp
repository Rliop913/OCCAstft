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
  return (Ra * Rb) + (Ia * Ib);
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

inline void DaCAdd(const int i_itr,
                   const unsigned int Half,
                   complex windowAdded[]) {
  unsigned int inRange = i_itr < Half;
  float Dpoint = windowAdded[i_itr].imag;
  float Apoint = windowAdded[i_itr + (Half * inRange)].imag;
  windowAdded[i_itr].imag = (Dpoint + Apoint) * inRange;
}

extern "C" void overlap_N_window(float * in,
                                 complex * buffer,
                                 const unsigned int & fullSize,
                                 const unsigned int & OFullSize,
                                 const int & windowSize,
                                 const unsigned int & OMove) {
#pragma omp parallel for
  for (unsigned int w_num = 0; w_num < OFullSize; w_num += 256) {
    for (int w_itr = 0; w_itr < 256; ++w_itr) {
      unsigned int FID = w_num + w_itr;
      unsigned int read_point = (unsigned int) ((FID) / windowSize) * OMove + ((FID) % windowSize);

      // printf("%u buffer overlap\n", OMove);//buffer[FID].imag);
      buffer[FID].real = read_point >= fullSize ? 0.0 : in[read_point];
      // * window_func((FID) % windowSize, windowSize);
      buffer[FID].imag = 0.0;
    }
  }
}

extern "C" void bitReverse_temp(complex * buffer,
                                complex * result,
                                const unsigned int & OFullSize,
                                const int & windowSize,
                                const int & radixData) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OFullSize; o_itr += 256) {
    for (int w_itr = 0; w_itr < 256; ++w_itr) {
      unsigned int Gidx = (o_itr + w_itr);
      unsigned int Lidx = (Gidx % windowSize);
      unsigned int dst_idx = reverseBits(Lidx, radixData);
      unsigned int BID = Gidx - Lidx + dst_idx;
      result[BID] = buffer[Gidx];
    }
  }
}

extern "C" void Butterfly(complex * buffer,
                          const int & windowSize,
                          const int & powed_stage,
                          const unsigned int & OHalfSize,
                          const int & radixData) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 256) {
    for (int i_itr = 0; i_itr < 256; ++i_itr) {
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

extern "C" void toPower(complex * buffer,
                        float * out,
                        const unsigned int & OHalfSize,
                        const int & windowRadix) {
  //toPower
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 256) {
    for (int i_itr = 0; i_itr < 256; ++i_itr) {
      const unsigned int GID = o_itr + i_itr;
      unsigned int BID = (GID >> (windowRadix - 1)) * (1 << windowRadix) + (GID & ((1 << (windowRadix - 1)) - 1));
      float powered = cmod(buffer[BID]);
      //powered = log10(powered);
      out[GID] = powered;
    }
  }
  return;
}

extern "C" void preprocessed_ODW10_STH_STFT(float * inData,
                                            const unsigned int & qtConst,
                                            const unsigned int & fullSize,
                                            const unsigned int & OMove,
                                            const unsigned int & OHalfSize,
                                            complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 512) {
    complex FBank[1024];
    complex SBank[1024];
    float windowAdded[512];
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (512)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 1024.0);
      FBank[i_itr].real *= window_func(i_itr, 1024);
      FBank[i_itr + 512].real -= (windowAdded[0] / 1024.0);
      FBank[i_itr + 512].real *= window_func(i_itr + 512, 1024);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 512), 1024);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 512), 1024);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 512), 1024);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 512), 1024);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 512), 1024);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 512), 1024);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 512), 1024);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 512), 1024);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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

extern "C" void preprocesses_ODW_10(float * inData,
                                    const unsigned int & qtConst,
                                    const unsigned int & fullSize,
                                    const unsigned int & OMove,
                                    complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < qtConst; ++o_itr) {
    complex windowBuffer[1024];
    float windowAdded[512];
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      unsigned int inRange = i_itr < 512;
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (512 * inRange)].real;
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 1024.0);
      windowBuffer[i_itr + 512].real -= (windowAdded[0] / 1024.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 1024);
      windowBuffer[i_itr + 512].real *= window_func(i_itr + 512, 1024);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      bufferOut[o_itr * 1024 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 1024 + i_itr + 512] = windowBuffer[i_itr + 512];
    }
  }
}

extern "C" void Stockhpotimized10(complex * buffer,
                                  const unsigned int & OHalfSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 512) {
    complex FBank[1024];
    complex SBank[1024];
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 512, 512), 1024);
      complex LEFT = buffer[o_itr * 2 + i_itr];
      complex RIGHT = buffer[o_itr * 2 + i_itr + 512];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 512), 1024);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 512), 1024);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 512), 1024);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 512), 1024);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 512), 1024);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 512), 1024);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 512), 1024);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 512] = storeR;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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

extern "C" void preprocesses_ODW_11(float * inData,
                                    const unsigned int & qtConst,
                                    const unsigned int & fullSize,
                                    const unsigned int & OMove,
                                    complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < qtConst; ++o_itr) {
    complex windowBuffer[2048];
    float windowAdded[1024];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (1024)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 2048.0);
      windowBuffer[i_itr + 1024].real -= (windowAdded[0] / 2048.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 2048);
      windowBuffer[i_itr + 1024].real *= window_func(i_itr + 1024, 2048);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      bufferOut[o_itr * 2048 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 2048 + i_itr + 1024] = windowBuffer[i_itr + 1024];
    }
  }
}

extern "C" void Stockhpotimized11(complex * buffer,
                                  const unsigned int & OHalfSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 1024) {
    complex FBank[2048];
    complex SBank[2048];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 1024), 2048);
      complex LEFT = buffer[o_itr * 2 + i_itr];
      complex RIGHT = buffer[o_itr * 2 + i_itr + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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

extern "C" void preprocessed_ODW11_STH_STFT(float * inData,
                                            const unsigned int & qtConst,
                                            const unsigned int & fullSize,
                                            const unsigned int & OMove,
                                            const unsigned int & OHalfSize,
                                            complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 1024) {
    complex FBank[2048];
    complex SBank[2048];
    float windowAdded[1024];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (1024)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 2048.0);
      FBank[i_itr].real *= window_func(i_itr, 2048);
      FBank[i_itr + 1024].real -= (windowAdded[0] / 2048.0);
      FBank[i_itr + 1024].real *= window_func(i_itr + 1024, 2048);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 1024, 1024), 2048);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 1024];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 256, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 255) | ((i_itr >> 8) << 9);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 256];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 128, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 127) | ((i_itr >> 7) << 8);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 128];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 64, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 63) | ((i_itr >> 6) << 7);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 64];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 32, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 31) | ((i_itr >> 5) << 6);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 32];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 15) | ((i_itr >> 4) << 5);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 16];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 7) | ((i_itr >> 3) << 4);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 8];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 4, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 3) | ((i_itr >> 2) << 3);
      complex LEFT = FBank[LeftIndex];
      complex RIGHT = FBank[LeftIndex + 4];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 2, 1024), 2048);
      unsigned int LeftIndex = (i_itr & 1) | ((i_itr >> 1) << 2);
      complex LEFT = SBank[LeftIndex];
      complex RIGHT = SBank[LeftIndex + 2];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      FBank[i_itr] = storeL;
      FBank[i_itr + 1024] = storeR;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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

extern "C" void preprocesses_ODW_12(float * inData,
                                    const unsigned int & qtConst,
                                    const unsigned int & fullSize,
                                    const unsigned int & OMove,
                                    complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < qtConst; ++o_itr) {
    complex windowBuffer[4096];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (2048)].real;
      windowBuffer[i_itr].imag = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (2048)].real;
      windowBuffer[i_itr].imag = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 512, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 256, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 128, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 64, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 32, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 16, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowBuffer);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1, windowBuffer);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      windowBuffer[i_itr].real -= (windowBuffer[0].imag / 4096.0);
      windowBuffer[i_itr + 2048].real -= (windowBuffer[0].imag / 4096.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 4096);
      windowBuffer[i_itr + 2048].real *= window_func(i_itr + 2048, 4096);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      windowBuffer[i_itr].real -= (windowBuffer[0].imag / 4096.0);
      windowBuffer[i_itr + 2048].real -= (windowBuffer[0].imag / 4096.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 4096);
      windowBuffer[i_itr + 2048].real *= window_func(i_itr + 2048, 4096);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 2048].imag = 0;
      bufferOut[o_itr * 4096 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 4096 + i_itr + 2048] = windowBuffer[i_itr + 2048];
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      windowBuffer[i_itr].imag = 0;
      windowBuffer[i_itr + 2048].imag = 0;
      bufferOut[o_itr * 4096 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 4096 + i_itr + 2048] = windowBuffer[i_itr + 2048];
      ;
    }
  }
}

extern "C" void Stockhpotimized12(complex * buffer,
                                  const unsigned int & OHalfSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 2048) {
    complex FBank[4096];
    complex SBank[4096];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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

extern "C" void preprocessed_ODW12_STH_STFT(float * inData,
                                            const unsigned int & qtConst,
                                            const unsigned int & fullSize,
                                            const unsigned int & OMove,
                                            const unsigned int & OHalfSize,
                                            complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 2048) {
    complex FBank[4096];
    complex SBank[4096];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (2048)].real;
      FBank[i_itr].imag = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (2048)].real;
      FBank[i_itr].imag = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1024, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1024, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 512, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 256, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 128, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 64, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 32, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 16, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, FBank);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1, FBank);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      FBank[i_itr].real -= (FBank[0].imag / 4096.0);
      FBank[i_itr].real *= window_func(i_itr, 4096);
      FBank[i_itr + 2048].real -= (FBank[0].imag / 4096.0);
      FBank[i_itr + 2048].real *= window_func(i_itr + 2048, 4096);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      FBank[i_itr].real -= (FBank[0].imag / 4096.0);
      FBank[i_itr].real *= window_func(i_itr, 4096);
      FBank[i_itr + 2048].real -= (FBank[0].imag / 4096.0);
      FBank[i_itr + 2048].real *= window_func(i_itr + 2048, 4096);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 2048), 4096);
      FBank[i_itr].imag = 0;
      FBank[i_itr + 2048].imag = 0;
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 2048, 2048), 4096);
      FBank[i_itr].imag = 0;
      FBank[i_itr + 2048].imag = 0;
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 2048];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 2048] = storeR;
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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

extern "C" void preprocesses_ODW_13(float * inData,
                                    const unsigned int & qtConst,
                                    const unsigned int & fullSize,
                                    const unsigned int & OMove,
                                    complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < qtConst; ++o_itr) {
    complex windowBuffer[8192];
    float windowAdded[4096];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 8192);
      windowBuffer[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 8192);
      windowBuffer[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 8192);
      windowBuffer[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 8192);
      windowBuffer[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      bufferOut[o_itr * 8192 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 8192 + i_itr + 4096] = windowBuffer[i_itr + 4096];
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      bufferOut[o_itr * 8192 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 8192 + i_itr + 4096] = windowBuffer[i_itr + 4096];
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      bufferOut[o_itr * 8192 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 8192 + i_itr + 4096] = windowBuffer[i_itr + 4096];
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      bufferOut[o_itr * 8192 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 8192 + i_itr + 4096] = windowBuffer[i_itr + 4096];
      ;
    }
  }
}

extern "C" void Stockhpotimized13(complex * buffer,
                                  const unsigned int & OHalfSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 4096) {
    complex FBank[8192];
    complex SBank[8192];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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

extern "C" void preprocessed_ODW13_STH_STFT(float * inData,
                                            const unsigned int & qtConst,
                                            const unsigned int & fullSize,
                                            const unsigned int & OMove,
                                            const unsigned int & OHalfSize,
                                            complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 4096) {
    complex FBank[8192];
    complex SBank[8192];
    float windowAdded[4096];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (4096)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr].real *= window_func(i_itr, 8192);
      FBank[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr].real *= window_func(i_itr, 8192);
      FBank[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr].real *= window_func(i_itr, 8192);
      FBank[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr].real *= window_func(i_itr, 8192);
      FBank[i_itr + 4096].real -= (windowAdded[0] / 8192.0);
      FBank[i_itr + 4096].real *= window_func(i_itr + 4096, 8192);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 4096, 4096), 8192);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 4096];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 4096] = storeR;
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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

extern "C" void preprocesses_ODW_14(float * inData,
                                    const unsigned int & qtConst,
                                    const unsigned int & fullSize,
                                    const unsigned int & OMove,
                                    complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < qtConst; ++o_itr) {
    complex windowBuffer[16384];
    float windowAdded[8192];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 16384);
      windowBuffer[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      bufferOut[o_itr * 16384 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 16384 + i_itr + 8192] = windowBuffer[i_itr + 8192];
      ;
    }
  }
}

extern "C" void Stockhpotimized14(complex * buffer,
                                  const unsigned int & OHalfSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 8192) {
    complex FBank[16384];
    complex SBank[16384];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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

extern "C" void preprocessed_ODW14_STH_STFT(float * inData,
                                            const unsigned int & qtConst,
                                            const unsigned int & fullSize,
                                            const unsigned int & OMove,
                                            const unsigned int & OHalfSize,
                                            complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 8192) {
    complex FBank[16384];
    complex SBank[16384];
    float windowAdded[8192];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (8192)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr].real *= window_func(i_itr, 16384);
      FBank[i_itr + 8192].real -= (windowAdded[0] / 16384.0);
      FBank[i_itr + 8192].real *= window_func(i_itr + 8192, 16384);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 8192, 8192), 16384);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 8192];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 8192] = storeR;
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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

extern "C" void preprocesses_ODW_15(float * inData,
                                    const unsigned int & qtConst,
                                    const unsigned int & fullSize,
                                    const unsigned int & OMove,
                                    complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < qtConst; ++o_itr) {
    complex windowBuffer[32768];
    float windowAdded[16384];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      float Dpoint = windowBuffer[i_itr].real;
      float Apoint = windowBuffer[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      windowBuffer[i_itr].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      windowBuffer[i_itr].real *= window_func(i_itr, 32768);
      windowBuffer[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      bufferOut[o_itr * 32768 + i_itr] = windowBuffer[i_itr];
      bufferOut[o_itr * 32768 + i_itr + 16384] = windowBuffer[i_itr + 16384];
      ;
    }
  }
}

extern "C" void Stockhpotimized15(complex * buffer,
                                  const unsigned int & OHalfSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 16384) {
    complex FBank[32768];
    complex SBank[32768];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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

extern "C" void preprocessed_ODW15_STH_STFT(float * inData,
                                            const unsigned int & qtConst,
                                            const unsigned int & fullSize,
                                            const unsigned int & OMove,
                                            const unsigned int & OHalfSize,
                                            complex * bufferOut) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 16384) {
    complex FBank[32768];
    complex SBank[32768];
    float windowAdded[16384];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //removing DC
      float Dpoint = FBank[i_itr].real;
      float Apoint = FBank[i_itr + (16384)].real;
      windowAdded[i_itr] = (Dpoint + Apoint);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 8192, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 4096, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 2048, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 1024, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 512, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 256, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 128, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 64, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 32, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 16, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 8, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 4, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 2, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      //DaCAdd(i_itr, 1, windowAdded);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      FBank[i_itr].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr].real *= window_func(i_itr, 32768);
      FBank[i_itr + 16384].real -= (windowAdded[0] / 32768.0);
      FBank[i_itr + 16384].real *= window_func(i_itr + 16384, 32768);
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
      complex thisTwiddle = twiddle(segmentK(i_itr, 16384, 16384), 32768);
      complex LEFT = FBank[i_itr];
      complex RIGHT = FBank[i_itr + 16384];
      complex storeL = cadd(LEFT, RIGHT);
      complex storeR = cmult(csub(LEFT, RIGHT), thisTwiddle);
      SBank[i_itr] = storeL;
      SBank[i_itr + 16384] = storeR;
      ;
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 1024; i_itr < 2048; ++i_itr) {
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
    for (int i_itr = 2048; i_itr < 3072; ++i_itr) {
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
    for (int i_itr = 3072; i_itr < 4096; ++i_itr) {
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
    for (int i_itr = 4096; i_itr < 5120; ++i_itr) {
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
    for (int i_itr = 5120; i_itr < 6144; ++i_itr) {
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
    for (int i_itr = 6144; i_itr < 7168; ++i_itr) {
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
    for (int i_itr = 7168; i_itr < 8192; ++i_itr) {
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
    for (int i_itr = 8192; i_itr < 9216; ++i_itr) {
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
    for (int i_itr = 9216; i_itr < 10240; ++i_itr) {
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
    for (int i_itr = 10240; i_itr < 11264; ++i_itr) {
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
    for (int i_itr = 11264; i_itr < 12288; ++i_itr) {
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
    for (int i_itr = 12288; i_itr < 13312; ++i_itr) {
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
    for (int i_itr = 13312; i_itr < 14336; ++i_itr) {
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
    for (int i_itr = 14336; i_itr < 15360; ++i_itr) {
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
    for (int i_itr = 15360; i_itr < 16384; ++i_itr) {
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

