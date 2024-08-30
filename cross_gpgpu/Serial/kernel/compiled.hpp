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

inline void DaCAdd(const int i_itr,
                   const unsigned int Half,
                   float windowAdded[]) {
  unsigned int inRange = i_itr < Half;
  float Dpoint = windowAdded[i_itr];
  float Apoint = windowAdded[i_itr + (Half * inRange)];
  windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
}

extern "C" void overlap_N_window(float * in,
                                 complex * buffer,
                                 const unsigned int & fullSize,
                                 const unsigned int & OFullSize,
                                 const int & windowSize,
                                 const unsigned int & OMove) {
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
                                            float * Rout,
                                            float * Iout) {
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 512) {
    float FRBank[1024];
    float FIBank[1024];
    float SRBank[1024];
    float SIBank[1024];
    float windowAdded[512];
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      float Dpoint = FRBank[i_itr];
      float Apoint = FRBank[i_itr + (512)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 256, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 128, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 64, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 32, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 16, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 8, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 4, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 2, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 1, windowAdded);
    }
    // for(int i_itr=0; i_itr < 512; ++i_itr; @inner)
    // {
    //     FRBank[i_itr] -= (windowAdded[0] / 1024.0);
    //     FRBank[i_itr] *= window_func(i_itr, 1024);
    //     FRBank[i_itr + 512] -= (windowAdded[0] / 1024.0);
    //     FRBank[i_itr + 512] *= window_func(i_itr + 512, 1024);
    // }

    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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

extern "C" void preprocesses_ODW_10(float * inData,
                                    const unsigned int & qtConst,
                                    const unsigned int & fullSize,
                                    const unsigned int & OMove,
                                    float * Rout) {
  for (unsigned int o_itr = 0; o_itr < qtConst; ++o_itr) {
    float wr[1024];
    float windowAdded[512];
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      unsigned int idx = o_itr * OMove + i_itr;
      unsigned int Ridx = o_itr * OMove + i_itr + 512;
      int isOverflowed = (idx < fullSize);
      int RisOverflowed = (Ridx < fullSize);
      idx *= isOverflowed;
      Ridx *= RisOverflowed;
      wr[i_itr] = inData[idx] * isOverflowed;
      wr[i_itr + 512] = inData[Ridx] * RisOverflowed;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      unsigned int inRange = i_itr < 512;
      float Dpoint = wr[i_itr];
      float Apoint = wr[i_itr + (512 * inRange)];
      windowAdded[i_itr] = (Dpoint + Apoint) * inRange;
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 256, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 128, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 64, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 32, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 16, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 8, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 4, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 2, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      DaCAdd(i_itr, 1, windowAdded);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      wr[i_itr] -= (windowAdded[0] / 1024.0);
      wr[i_itr + 512] -= (windowAdded[0] / 1024.0);
      wr[i_itr] *= window_func(i_itr, 1024);
      wr[i_itr + 512] *= window_func(i_itr + 512, 1024);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      Rout[o_itr * 1024 + i_itr] = wr[i_itr];
      Rout[o_itr * 1024 + i_itr + 512] = wr[i_itr + 512];
    }
  }
}

extern "C" void Stockhpotimized10(float * Rout,
                                  float * Iout,
                                  const unsigned int & OHalfSize) {
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 512) {
    float FRBank[1024];
    float FIBank[1024];
    float SRBank[1024];
    float SIBank[1024];
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
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

extern "C" void preprocesses_ODW_11(float * inData,
                                    const unsigned int & qtConst,
                                    const unsigned int & fullSize,
                                    const unsigned int & OMove,
                                    float * Rout) {
  for (unsigned int o_itr = 0; o_itr < qtConst; ++o_itr) {
    float wr[2048];
    float windowAdded[1024];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      float Dpoint = wr[i_itr];
      float Apoint = wr[i_itr + (1024)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 512, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 256, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 128, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 64, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 32, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 16, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 8, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 4, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 2, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 1, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      wr[i_itr] -= (windowAdded[0] / 2048.0);
      wr[i_itr + 1024] -= (windowAdded[0] / 2048.0);
      wr[i_itr] *= window_func(i_itr, 2048);
      wr[i_itr + 1024] *= window_func(i_itr + 1024, 2048);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      Rout[o_itr * 2048 + i_itr] = wr[i_itr];
      Rout[o_itr * 2048 + i_itr + 1024] = wr[i_itr + 1024];
    }
  }
}

extern "C" void Stockhpotimized11(float * Rout,
                                  float * Iout,
                                  const unsigned int & OHalfSize) {
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 1024) {
    float FRBank[2048];
    float FIBank[2048];
    float SRBank[2048];
    float SIBank[2048];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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

extern "C" void preprocessed_ODW11_STH_STFT(float * inData,
                                            const unsigned int & qtConst,
                                            const unsigned int & fullSize,
                                            const unsigned int & OMove,
                                            const unsigned int & OHalfSize,
                                            float * Rout,
                                            float * Iout) {
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 1024) {
    float FRBank[2048];
    float FIBank[2048];
    float SRBank[2048];
    float SIBank[2048];
    float windowAdded[1024];
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      float Dpoint = FRBank[i_itr];
      float Apoint = FRBank[i_itr + (1024)];
      windowAdded[i_itr] = (Dpoint + Apoint);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 512, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 256, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 128, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 64, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 32, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 16, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 8, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 4, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 2, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      DaCAdd(i_itr, 1, windowAdded);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
      FRBank[i_itr] -= (windowAdded[0] / 2048.0);
      FRBank[i_itr] *= window_func(i_itr, 2048);
      FRBank[i_itr + 1024] -= (windowAdded[0] / 2048.0);
      FRBank[i_itr + 1024] *= window_func(i_itr + 1024, 2048);
    }
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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
    for (int i_itr = 0; i_itr < 1024; ++i_itr) {
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

extern "C" void Overlap_Common(float * inData,
                               const unsigned int & OFullSize,
                               const unsigned int & fullSize,
                               const unsigned int & windowRadix,
                               const unsigned int & OMove,
                               float * outReal) {
  for (unsigned int o_itr = 0; o_itr < OFullSize; o_itr += 1024) {
    for (unsigned int i_itr = 0; i_itr < 1024; ++i_itr) {
      const unsigned int overlapIdx = o_itr + i_itr;
      const unsigned int windowIdx = (overlapIdx >> windowRadix);
      const unsigned int windowLocalIdx = overlapIdx & ((1 << windowRadix) - 1);
      const unsigned int originIdx = windowIdx * OMove + windowLocalIdx;
      const unsigned int exceeded = originIdx < fullSize;
      outReal[overlapIdx] = inData[originIdx * exceeded] * exceeded;
    }
  }
}

extern "C" void Window_Common(float * outReal,
                              const unsigned int & OFullSize,
                              const unsigned int & windowRadix) {
  for (unsigned int o_itr = 0; o_itr < OFullSize; o_itr += 1024) {
    for (unsigned int i_itr = 0; i_itr < 1024; ++i_itr) {
      unsigned int Gidx = o_itr + i_itr;
      outReal[Gidx] *= window_func((Gidx & (windowRadix - 1)), 1 << windowRadix);
    }
  }
}

extern "C" void DCRemove_Common(float * outReal,
                                const unsigned int & OFullSize,
                                const unsigned int & windowSize) {
  for (unsigned int o_itr = 0; o_itr < OFullSize; o_itr += windowSize) {
    float added[128];
    //for removing DC
    for (unsigned int windowItr = 0; windowItr < windowSize; windowItr += 64) {
      for (unsigned int i_itr = 0; i_itr < 64; ++i_itr) {
        added[i_itr + 64] = outReal[o_itr + windowItr + i_itr];
      }
      for (unsigned int i_itr = 0; i_itr < 64; ++i_itr) {
        added[i_itr] += added[i_itr + 64];
      }
    }
    for (unsigned int segment = 32; segment > 0; segment >>= 1) {
      for (unsigned int i_itr = 0; i_itr < 64; ++i_itr) {
        unsigned int inSegment = i_itr < segment;
        float left = added[i_itr];
        float right = added[i_itr + segment];
        added[i_itr] = (left + right) * inSegment;
      }
    }
    for (unsigned int windowItr = 0; windowItr < windowSize; windowItr += 64) {
      for (unsigned int i_itr = 0; i_itr < 64; ++i_itr) {
        outReal[o_itr + windowItr + i_itr] -= (added[0] / (float) windowSize);
      }
    }
  }
}

extern "C" void StockHamDITCommon(float * inReal,
                                  float * inImag,
                                  float * outReal,
                                  float * outImag,
                                  const unsigned int & HwindowSize,
                                  const unsigned int & stageRadix,
                                  const unsigned int & OHalfSize,
                                  const unsigned int & radixData) {
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 256) {
    for (unsigned int i_itr = 0; i_itr < 256; ++i_itr) {
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

