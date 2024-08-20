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

__device__ inline spair stockhamIndexer(const int localIdx,
                                        const int segmentSize,
                                        const unsigned int HalfWinSize) {
  spair res;
  res.lload = localIdx + (localIdx & (~(segmentSize - 1)));
  res.rload = res.lload + segmentSize;
  res.lsave = localIdx;
  res.rsave = HalfWinSize + res.lsave;
  return res;
}


// pairs
// indexing(const unsigned int ID,const int powed_stage)
// {
//     pairs temp;
//     temp.first = ID;
//     temp.second = ID + (ID % (powed_stage*2) >= powed_stage ? -powed_stage : powed_stage);
//     return temp;
// }

__device__ inline int calculateK(int windowIDX,
                                 int powed_stage,
                                 int windowSize) {
  return ((windowIDX % powed_stage) * windowSize) / (powed_stage * 2);
}

__device__ inline int segmentK(const int lsave,
                               const int segmentSize,
                               const int HwindowSize) {
  // return lsave;
  // return calculateK(lsave, segmentSize, windowSize);
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

extern "C" __global__ __launch_bounds__(256) void _occa_removeDC_0(complex * buffer,
                                                                   const unsigned int OFullSize,
                                                                   float * qt_buffer,
                                                                   const int windowSize) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int IDX = o_itr + i_itr;
      atomicAdd(&qt_buffer[IDX / windowSize], buffer[IDX].imag);
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void _occa_removeDC_1(complex * buffer,
                                                                   const unsigned int OFullSize,
                                                                   float * qt_buffer,
                                                                   const int windowSize) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int IDX = o_itr + i_itr;
      buffer[IDX].imag -= (qt_buffer[IDX / windowSize] / (float) windowSize);
    }
  }
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

extern "C" __global__ __launch_bounds__(256) void _occa_overlap_N_window_imag_0(float * in,
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
      buffer[FID].imag = read_point >= fullSize ? 0.0 : in[read_point] * window_func(
        (FID) % windowSize,
        windowSize
      );
      buffer[FID].real = 0.0;
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

extern "C" __global__ __launch_bounds__(256) void _occa_bitReverse_0(complex * buffer,
                                                                     const unsigned int OFullSize,
                                                                     const int windowSize,
                                                                     const int radixData) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int w_itr = 0 + threadIdx.x;
      unsigned int dst_idx = reverseBits(((o_itr + w_itr) % windowSize), radixData);
      unsigned int BID = o_itr + w_itr - ((o_itr + w_itr) % windowSize) + dst_idx;
      buffer[BID].real = buffer[o_itr + w_itr].imag;
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void _occa_endPreProcess_0(complex * buffer,
                                                                        const unsigned int OFullSize) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int i_itr = 0 + threadIdx.x;
      // printf("%f data buffer\n", buffer[o_itr + i_itr].real);
      buffer[o_itr + i_itr].imag = 0.0;
    }
  }
}

extern "C" __global__ __launch_bounds__(512) void _occa_StockhamButterfly10_0(complex * buffer,
                                                                              const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (512 * blockIdx.x);
    __shared__ complex bank_first[1024];
    __shared__ complex bank_second[1024];
    {
      int i_itr = 0 + threadIdx.x;
      unsigned int global_idx = (o_itr + i_itr);
      pairs idx = indexer(global_idx, 512);
      spair lidx = stockhamIndexer(i_itr, 512, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 512, 512), 1024);
      complex cfirst = buffer[idx.first];
      complex csecond = buffer[idx.second];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      spair lidx = stockhamIndexer(i_itr, 256, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 256, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      bank_first[lidx.lsave] = cadd(cfirst, csecond);
      bank_first[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      spair lidx = stockhamIndexer(i_itr, 128, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 128, 512), 1024);
      complex cfirst = bank_first[lidx.lload];
      complex csecond = bank_first[lidx.rload];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      spair lidx = stockhamIndexer(i_itr, 64, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 64, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      bank_first[lidx.lsave] = cadd(cfirst, csecond);
      bank_first[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      spair lidx = stockhamIndexer(i_itr, 32, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 32, 512), 1024);
      complex cfirst = bank_first[lidx.lload];
      complex csecond = bank_first[lidx.rload];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      spair lidx = stockhamIndexer(i_itr, 16, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 16, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      bank_first[lidx.lsave] = cadd(cfirst, csecond);
      bank_first[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      spair lidx = stockhamIndexer(i_itr, 8, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 8, 512), 1024);
      complex cfirst = bank_first[lidx.lload];
      complex csecond = bank_first[lidx.rload];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      spair lidx = stockhamIndexer(i_itr, 4, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 4, 512), 1024);
      complex cfirst = bank_second[lidx.lload];
      complex csecond = bank_second[lidx.rload];
      bank_first[lidx.lsave] = cadd(cfirst, csecond);
      bank_first[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
      spair lidx = stockhamIndexer(i_itr, 2, 512);
      complex thisTwiddle = twiddle(segmentK(lidx.lsave, 2, 512), 1024);
      complex cfirst = bank_first[lidx.lload];
      complex csecond = bank_first[lidx.rload];
      bank_second[lidx.lsave] = cadd(cfirst, csecond);
      bank_second[lidx.rsave] = cmult(csub(cfirst, csecond), thisTwiddle);
    }
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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

    // for(int i_itr = 0; i_itr < 512; ++i_itr; @inner)
    // {
    //     pairs GIDX;
    //     GIDX.first = o_itr * 2 + i_itr;
    //     GIDX.second= GIDX.first + 512;
    //     buffer[GIDX.first] = bank_first[i_itr];
    //     buffer[GIDX.second] = bank_first[i_itr + 512];
    // }
  }
}


// printf("%d STKstage k = %d\n", POWSTAGE, calculateK(idx.first, POWSTAGE, WINSIZE));

extern "C" __global__ __launch_bounds__(512) void _occa_OptimizedDIFButterfly10_0(complex * buffer,
                                                                                  const unsigned int OHalfSize) {
  {
    unsigned int o_itr = 0 + (512 * blockIdx.x);
    __shared__ complex bank_first[1024];
    __shared__ complex bank_second[1024];
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    __syncthreads();
    {
      int i_itr = 0 + threadIdx.x;
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
    // for(int i_itr = 0; i_itr < 256; ++i_itr; @inner)
    // {
    //     @barrier();
    //     pairs idx = indexer((i_itr + o_itr) % (1024 / 2), 512);
    //     complex cfirst = bank_second[idx.first];
    //     complex csecond = bank_second[idx.second];

    //     complex this_twiddle = twiddle(calculateK(idx.first, 512, 1024), 1024);
    //     complex tempcplx = cmult(csecond, this_twiddle);

    //     pairs Gidx = indexer(i_itr + o_itr, 512);
    //     buffer[Gidx.first] = cadd(cfirst, tempcplx);
    //     buffer[Gidx.second] = csub(cfirst, tempcplx);

    //     bank_first[idx.first] = cadd(cfirst, tempcplx);
    //     bank_first[idx.second] = csub(cfirst, tempcplx);
    // }
    // for(int i_itr = 0; i_itr < 256; ++i_itr; @inner)
    // {
    //     pairs idx = indexer(i_itr, 256);
    //     complex cfirst = bank_first[idx.first];
    //     complex csecond = bank_first[idx.second];

    //     complex this_twiddle = twiddle(calculateK(idx.first, 256, 1024), 1024);
    //     complex tempcplx = cmult(csecond, this_twiddle);
    //     pairs GIDX = indexer(o_itr + i_itr, 256);
    //     buffer[GIDX.first] = cadd(cfirst, tempcplx);
    //     buffer[GIDX.second] = csub(cfirst, tempcplx);
    // }
    // for(int i_itr = 0; i_itr < 256; ++i_itr; @inner)
    // {
    //     pairs LIDX = indexer((i_itr + o_itr) % 512, 512);
    //     pairs GIDX = indexer(i_itr + o_itr, 512);
    //     // o_itr = 432153245;
    //     complex cfirst = buffer[GIDX.first];
    //     complex csecond = buffer[GIDX.second];
    //     // printf("GDIX F : %d, S : %d\n", GIDX.first, GIDX.second);
    //     // printf("stk=%d\n", calculateK(LIDX.first, 512, 1024));
    //     complex this_twiddle = twiddle(calculateK(GIDX.first, 512, 1024), 1024);
    //     complex tempcplx = cmult(csecond, this_twiddle);
    //     printf("%f , %f idx: %d idxS: %d\n", csecond.real, cfirst.real, GIDX.first, GIDX.second);
    //     buffer[GIDX.first] = cadd(cfirst, tempcplx);
    //     buffer[GIDX.second] = csub(cfirst, tempcplx);
    // }

  }
}


// quot = (fullSize / overlap_ratio) / overlap_ratio / window_size
//calculateK(int low_in_window, int powed_stage, int windowSize)

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

