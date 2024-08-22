#define _USE_MATH_DEFINES
#include <math.h>
// #include <math.h>
// #include <cstdio>

typedef struct complex_t {
  float real, imag;
} complex;

typedef struct pairs_t {
  unsigned int first, second;
} pairs;

typedef struct stockhamPairs {
  unsigned int lload, rload, lsave, rsave;
} spair;

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

inline spair stockhamIndexer(const int localIdx,
                             const int segmentSize,
                             const unsigned int HalfWinSize) {
  spair res;
  res.lload = localIdx + (localIdx & (~(segmentSize - 1)));
  res.rload = res.lload + segmentSize;
  res.lsave = localIdx;
  res.rsave = HalfWinSize + res.lsave;
  return res;
}

typedef struct futureIDX {
  unsigned int Fidx, isR;
} FIDX;

inline FIDX GetNextIndex(const int localIdx,
                         const int nextSegment,
                         const int NsegmentRadix) {
  FIDX fidx;
  fidx.isR = (localIdx & nextSegment) >> (NsegmentRadix - 1);
  unsigned int temp = fidx.isR;
  unsigned int isL = (temp == 0);
  fidx.Fidx = ((localIdx & (nextSegment - 1)) + ((localIdx >> (NsegmentRadix + 1)) << NsegmentRadix)) * ((isL)) + (((localIdx - nextSegment) & (nextSegment - 1)) + (((localIdx - nextSegment) >> (NsegmentRadix + 1)) << NsegmentRadix)) * temp;
  fidx.isR = ((temp != 0));
  printf("%u unsigned \n", fidx.isR);
  return fidx;
}


// pairs
// indexing(const unsigned int ID,const int powed_stage)
// {
//     pairs temp;
//     temp.first = ID;
//     temp.second = ID + (ID % (powed_stage*2) >= powed_stage ? -powed_stage : powed_stage);
//     return temp;
// }

inline int calculateK(int windowIDX,
                      int powed_stage,
                      int windowSize) {
  return ((windowIDX % powed_stage) * windowSize) / (powed_stage * 2);
}

inline int segmentK(const int lsave,
                    const int segmentSize,
                    const int HwindowSize) {
  // return lsave;
  // return calculateK(lsave, segmentSize, windowSize);
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

extern "C" void removeDC(complex * buffer,
                         const unsigned int & OFullSize,
                         float * qt_buffer,
                         const int & windowSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OFullSize; o_itr += 256) {
    for (int i_itr = 0; i_itr < 256; ++i_itr) {
      unsigned int IDX = o_itr + i_itr;
#pragma omp atomic
      qt_buffer[IDX / windowSize] += buffer[IDX].imag;
    }
  }
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OFullSize; o_itr += 256) {
    for (int i_itr = 0; i_itr < 256; ++i_itr) {
      unsigned int IDX = o_itr + i_itr;
      buffer[IDX].imag -= (qt_buffer[IDX / windowSize] / (float) windowSize);
    }
  }
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

extern "C" void overlap_N_window_imag(float * in,
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
      buffer[FID].imag = read_point >= fullSize ? 0.0 : in[read_point] * window_func(
        (FID) % windowSize,
        windowSize
      );
      buffer[FID].real = 0.0;
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

extern "C" void bitReverse(complex * buffer,
                           const unsigned int & OFullSize,
                           const int & windowSize,
                           const int & radixData) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OFullSize; o_itr += 256) {
    for (int w_itr = 0; w_itr < 256; ++w_itr) {
      unsigned int dst_idx = reverseBits(((o_itr + w_itr) % windowSize), radixData);
      unsigned int BID = o_itr + w_itr - ((o_itr + w_itr) % windowSize) + dst_idx;
      buffer[BID].real = buffer[o_itr + w_itr].imag;
    }
  }
}

extern "C" void endPreProcess(complex * buffer,
                              const unsigned int & OFullSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OFullSize; o_itr += 256) {
    for (int i_itr = 0; i_itr < 256; ++i_itr) {
      // printf("%f data buffer\n", buffer[o_itr + i_itr].real);
      buffer[o_itr + i_itr].imag = 0.0;
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

extern "C" void StockhamButterfly10(complex * buffer,
                                    const unsigned int & OHalfSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 512) {
    complex bank_first[1024];
    complex bank_second[1024];
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      unsigned int global_idx = (o_itr + i_itr);
      pairs idx = indexer(global_idx, 512);
      spair lidx = stockhamIndexer(i_itr, 512, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 512, 512), 1024);
      complex cfirst = buffer[idx.first];
      complex csecond = buffer[idx.second];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      spair lidx = stockhamIndexer(i_itr, 256, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 256, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      bank_first[lidx.lsave] = cadd(cfirst, csecond);
      bank_first[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      spair lidx = stockhamIndexer(i_itr, 128, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 128, 512), 1024);
      complex cfirst = bank_first[lidx.lload];
      complex csecond = bank_first[lidx.rload];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      spair lidx = stockhamIndexer(i_itr, 64, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 64, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      bank_first[lidx.lsave] = cadd(cfirst, csecond);
      bank_first[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      spair lidx = stockhamIndexer(i_itr, 32, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 32, 512), 1024);
      complex cfirst = bank_first[lidx.lload];
      complex csecond = bank_first[lidx.rload];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      spair lidx = stockhamIndexer(i_itr, 16, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 16, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      bank_first[lidx.lsave] = cadd(cfirst, csecond);
      bank_first[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      spair lidx = stockhamIndexer(i_itr, 8, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 8, 512), 1024);
      complex cfirst = bank_first[lidx.lload];
      complex csecond = bank_first[lidx.rload];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      spair lidx = stockhamIndexer(i_itr, 4, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 4, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      bank_first[lidx.lsave] = cadd(cfirst, csecond);
      bank_first[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      spair lidx = stockhamIndexer(i_itr, 2, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 2, 512), 1024);
      complex cfirst = bank_first[lidx.lload];
      complex csecond = bank_first[lidx.rload];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs GIDX;
      GIDX.first = o_itr * 2 + i_itr;
      GIDX.second = GIDX.first + 512;
      spair lidx = stockhamIndexer(i_itr, 1, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 1, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      buffer[GIDX.first] = cadd(cfirst, csecond);
      buffer[GIDX.second] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
  }
}

extern "C" void OptimizedDIFButterfly10(complex * buffer,
                                        const unsigned int & OHalfSize) {
#pragma omp parallel for
  for (unsigned int o_itr = 0; o_itr < OHalfSize; o_itr += 512) {
    complex bank_first[1024];
    complex bank_second[1024];
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      int global_idx = (o_itr + i_itr);
      pairs idx = indexer(global_idx, 512);
      pairs Lidx = indexer((i_itr + o_itr) % 512, 512);
      complex cfirst = buffer[idx.first];
      complex csecond = buffer[idx.second];
      complex this_twiddle = twiddle(calculateK(i_itr, 512, 1024), 1024);

      // printf("first: %f second: %f  idx: %u\n", cfirst.real, csecond.real, idx.first);

      bank_second[Lidx.first] = cadd(cfirst, csecond);
      bank_second[Lidx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      //debug
      buffer[idx.first] = bank_second[Lidx.first];
      buffer[idx.second] = bank_second[Lidx.second];
      //debug
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 256);
      complex cfirst = bank_second[idx.first];
      complex csecond = bank_second[idx.second];
      complex this_twiddle = twiddle(
        calculateK(idx.first, 256, 1024),
        1024
      );
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_first[idx.first] = cadd(cfirst, csecond);
      bank_first[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 256);
      buffer[Gidx.first] = bank_first[idx.first];
      buffer[Gidx.second] = bank_first[idx.second];
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 128);
      complex cfirst = bank_first[idx.first];
      complex csecond = bank_first[idx.second];
      complex this_twiddle = twiddle(
        calculateK(idx.first, 128, 1024),
        1024
      );
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_second[idx.first] = cadd(cfirst, csecond);
      bank_second[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 128);
      buffer[Gidx.first] = bank_second[idx.first];
      buffer[Gidx.second] = bank_second[idx.second];
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 64);
      complex cfirst = bank_second[idx.first];
      complex csecond = bank_second[idx.second];
      complex this_twiddle = twiddle(
        calculateK(idx.first, 64, 1024),
        1024
      );
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_first[idx.first] = cadd(cfirst, csecond);
      bank_first[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 64);
      buffer[Gidx.first] = bank_first[idx.first];
      buffer[Gidx.second] = bank_first[idx.second];
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 32);
      complex cfirst = bank_first[idx.first];
      complex csecond = bank_first[idx.second];
      complex this_twiddle = twiddle(
        calculateK(idx.first, 32, 1024),
        1024
      );
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_second[idx.first] = cadd(cfirst, csecond);
      bank_second[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 32);
      buffer[Gidx.first] = bank_second[idx.first];
      buffer[Gidx.second] = bank_second[idx.second];
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 16);
      complex cfirst = bank_second[idx.first];
      complex csecond = bank_second[idx.second];
      complex this_twiddle = twiddle(
        calculateK(idx.first, 16, 1024),
        1024
      );
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_first[idx.first] = cadd(cfirst, csecond);
      bank_first[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 16);
      buffer[Gidx.first] = bank_first[idx.first];
      buffer[Gidx.second] = bank_first[idx.second];
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 8);
      complex cfirst = bank_first[idx.first];
      complex csecond = bank_first[idx.second];
      complex this_twiddle = twiddle(calculateK(idx.first, 8, 1024), 1024);
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_second[idx.first] = cadd(cfirst, csecond);
      bank_second[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 8);
      buffer[Gidx.first] = bank_second[idx.first];
      buffer[Gidx.second] = bank_second[idx.second];
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 4);
      complex cfirst = bank_second[idx.first];
      complex csecond = bank_second[idx.second];
      complex this_twiddle = twiddle(calculateK(idx.first, 4, 1024), 1024);
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_first[idx.first] = cadd(cfirst, csecond);
      bank_first[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 4);
      buffer[Gidx.first] = bank_first[idx.first];
      buffer[Gidx.second] = bank_first[idx.second];
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 2);
      complex cfirst = bank_first[idx.first];
      complex csecond = bank_first[idx.second];
      complex this_twiddle = twiddle(calculateK(idx.first, 2, 1024), 1024);
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_second[idx.first] = cadd(cfirst, csecond);
      bank_second[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 2);
      buffer[Gidx.first] = bank_second[idx.first];
      buffer[Gidx.second] = bank_second[idx.second];
    }
    for (int i_itr = 0; i_itr < 512; ++i_itr) {
      pairs idx = indexer((i_itr + o_itr) % (512), 1);
      complex cfirst = bank_second[idx.first];
      complex csecond = bank_second[idx.second];
      complex this_twiddle = twiddle(calculateK(idx.first, 1, 1024), 1024);
      complex tempcplx = cmult(csecond, this_twiddle);
      bank_first[idx.first] = cadd(cfirst, csecond);
      bank_first[idx.second] = cmult(csub(cfirst, csecond), this_twiddle);
      pairs Gidx = indexer(i_itr + o_itr, 1);
      buffer[Gidx.first] = bank_first[idx.first];
      buffer[Gidx.second] = bank_first[idx.second];
    }
  }
}


// quot = (fullSize / overlap_ratio) / overlap_ratio / window_size
//calculateK(int low_in_window, int powed_stage, int windowSize)

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
      // 12 34 56 78
      //(windowSize/powed_stage)
      int k = (GID % powed_stage) * (windowSize / (2 * powed_stage));
      //(GID%powed_stage) * (windowSize / powed_stage);
      complex this_twiddle = twiddle(k, windowSize);
      // printf("%f -- %f <gpu>> %d\n", this_twiddle.real, this_twiddle.imag, k);
      // printf("%f -- %f\n", buffer[butterfly_target.first].real, buffer[butterfly_target.second].real);
      //complex tempcplx = cmult(buffer[butterfly_target.second], this_twiddle);
      // if(powed_stage == 512)
      // {
      //     printf("%f , %f -- %d __ %d\n", buffer[butterfly_target.first].real, buffer[butterfly_target.second].real, butterfly_target.first, butterfly_target.second);

      // }
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

