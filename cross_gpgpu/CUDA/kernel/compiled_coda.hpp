

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

__device__ pairs indexer(const unsigned int ID,
                         const int powed_stage) {
  pairs temp;
  temp.first = ID + (ID & (~(powed_stage - 1)));
  temp.second = temp.first + powed_stage;
  return temp;
}

__device__ int calculateK(int windowIDX,
                          int powed_stage,
                          int windowSize) {
  int position = windowIDX;
  // 그룹 내에서의 위치
  int k = (position * windowSize) / powed_stage;
  return k;
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
                                                                   unsigned int OFullSize,
                                                                   float * qt_buffer,
                                                                   int windowSize) {
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
                                                                   unsigned int OFullSize,
                                                                   float * qt_buffer,
                                                                   int windowSize) {
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
                                                                           unsigned int fullSize,
                                                                           const unsigned int OFullSize,
                                                                           const int windowSize,
                                                                           unsigned int OMove) {
  {
    unsigned int w_num = 0 + (256 * blockIdx.x);
    {
      int w_itr = 0 + threadIdx.x;
      unsigned int FID = w_num + w_itr;
      unsigned int read_point = (int) ((FID) / windowSize) * OMove + ((FID) % windowSize);
      buffer[FID].imag = read_point >= fullSize ? 0.0 : in[read_point] * window_func(
        (FID) % windowSize,
        windowSize
      );
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void _occa_bitReverse_0(complex * buffer,
                                                                     const unsigned int OFullSize,
                                                                     int windowSize,
                                                                     int radixData) {
  {
    unsigned int o_itr = 0 + (256 * blockIdx.x);
    {
      int w_itr = 0 + threadIdx.x;
      unsigned int dst_idx = reverseBits((o_itr + w_itr % windowSize), radixData);
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
      buffer[o_itr + i_itr].imag = 0.0;
    }
  }
}


// quot = (fullSize / overlap_ratio) / overlap_ratio / window_size
//calculateK(int low_in_window, int powed_stage, int windowSize)

extern "C" __global__ __launch_bounds__(256) void _occa_Butterfly_0(complex * buffer,
                                                                    int windowSize,
                                                                    const int powed_stage,
                                                                    const unsigned int OHalfSize,
                                                                    int radixData) {
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
      //printf("%f -- %f <gpu>> %d\n", this_twiddle.real, this_twiddle.imag, k);
      //printf("%f -- %f\n", buffer[butterfly_target.first].real, buffer[butterfly_target.second].real);
      complex tempcplx = cmult(
        buffer[butterfly_target.second],
        this_twiddle
      );
      //printf("%f -- %f\n", tempcplx.real, tempcplx.imag);
      complex tempx = cadd(buffer[butterfly_target.first], tempcplx);
      complex tempy = csub(buffer[butterfly_target.first], tempcplx);
      buffer[butterfly_target.first] = tempx;
      buffer[butterfly_target.second] = tempy;
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void _occa_toPower_0(complex * buffer,
                                                                  float * out,
                                                                  const unsigned int OHalfSize,
                                                                  int windowRadix) {
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

