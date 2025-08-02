#include <torch/types.h>

/// q: (L, d)
/// k: (L, d)
/// v: (L, d)
/// o: (L, d)
/// number_of_thread_blocks: L / Q_BLOCK_SIZE
///
/// A thread in the block will compute a single row in the o. Therefore, we do
/// not have outer loop. The thread block contains Q_BLOCK_SIZE threads.
/// KV_BLOCK_SIZE must be multiple of Q_BOCK_SIZE
template <int Q_BLOCK_SIZE, int KV_BLOCK_SIZE, int HEAD_DIM>
__global__ void
flash_attention(const float *__restrict__ q, const float *__restrict__ k,
                const float *__restrict__ v, float *__restrict__ o,
                const int length, float softmax_scale) {
  __shared__ float q_shared[Q_BLOCK_SIZE][HEAD_DIM];
  __shared__ float k_shared[KV_BLOCK_SIZE][HEAD_DIM];
  /// Transpose v to increase memory access affinity
  __shared__ float v_shared[HEAD_DIM][KV_BLOCK_SIZE];
  __shared__ float o_shared[Q_BLOCK_SIZE][HEAD_DIM];

  constexpr int NUM_THREADS = Q_BLOCK_SIZE;
  constexpr auto NUM_KV_ELEMENTS_PER_THREAD = KV_BLOCK_SIZE / Q_BLOCK_SIZE;
  const auto row_index = blockIdx.x * Q_BLOCK_SIZE + threadIdx.x;
  const auto q_element_start_ptr = q + row_index * HEAD_DIM;
  const auto o_element_start_ptr = o + row_index * HEAD_DIM;

  float m_old = -INFINITY;
  float m_new = -INFINITY;
  float denominator = 0.0;
  float attentions[KV_BLOCK_SIZE] = {0.0};
  const auto num_iters = length / KV_BLOCK_SIZE;

  /// Load Q_BLOCK into shared memory and set o_shared to 0.0
  /// TODO: vector load
  for (int i = 0; i < HEAD_DIM; ++i) {
    o_shared[threadIdx.x][i] = 0.0;
    q_shared[threadIdx.x][i] = q_element_start_ptr[i];
  }

  /// Loop over k/v
  for (int kv_iter = 0; kv_iter < num_iters; ++kv_iter) {
    /// Load blocked k/v into shared memory
    {
      /// TODO: vector load
#pragma unroll
      for (int load_iter = 0; load_iter < NUM_KV_ELEMENTS_PER_THREAD;
           ++load_iter) {
        /// Destion
        const auto k_shared_start_offset =
            (load_iter * NUM_THREADS + threadIdx.x) * HEAD_DIM;
        const auto k_shared_start_ptr = (&**k_shared) + k_shared_start_offset;
        const auto v_shared_start_ptr =
            (&**v_shared) + load_iter * NUM_THREADS + threadIdx.x;

        /// Source
        const auto kv_start_offset =
            (kv_iter * KV_BLOCK_SIZE + load_iter * NUM_THREADS + threadIdx.x) *
            HEAD_DIM;
        const auto k_start_ptr = k + kv_start_offset;
        const auto v_start_ptr = v + kv_start_offset;

        for (int i = 0; i < HEAD_DIM; ++i) {
          k_shared_start_ptr[i] = k_start_ptr[i];
          v_shared_start_ptr[i * KV_BLOCK_SIZE] = v_start_ptr[i];
        }

        /// Ensure k/v is loaded. All of the threads in the thread block will
        /// view the correct k/v
        __syncthreads();
      }
    }

    /// Compute q @ k
    /// TODO: It is pretty slow
    /// TODO: thread tiling with outer product
    for (int i = 0; i < KV_BLOCK_SIZE; ++i) {
      float &a = attentions[i];
      a = 0.0f;
      for (int j = 0; j < HEAD_DIM; ++j)
        a += q_shared[threadIdx.x][j] * k_shared[i][j];
      a *= softmax_scale;
      m_new = m_new > a ? m_new : a;
    }

    /// Compute unscaled attention
    float s = 0.0;
    for (int i = 0; i < KV_BLOCK_SIZE; ++i) {
      auto a = __expf(attentions[i] - m_new);
      s += a;
      attentions[i] = a;
    }

    /// Compute unscaled denominator
    auto scale_diff = kv_iter == 0 ? 1.0f : __expf(m_old - m_new);
    denominator = scale_diff * denominator + s;

    /// Compute output + attention @ V
    for (int j = 0; j < HEAD_DIM; ++j) {
      auto scaled_o = o_shared[threadIdx.x][j] * scale_diff;
      for (int i = 0; i < KV_BLOCK_SIZE; ++i) {
        scaled_o += attentions[i] * v_shared[j][i];
      }
      o_shared[threadIdx.x][j] = scaled_o;
    }

    /// In next iteration, m_new is old now. And we should reset attentions
    m_old = m_new;

    /// Sync threads such that new load will happen after all threads within
    /// thread block have finished the computation
    __syncthreads();
  }

  /// Scale the output with denominator and write to HBM
  for (int i = 0; i < HEAD_DIM; ++i)
    o_element_start_ptr[i] = o_shared[threadIdx.x][i] / denominator;
}

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  auto o = torch::zeros_like(q);
  constexpr int Q_BLOCK_SIZE = 32;
  constexpr int KV_BLOCK_SIZE = 32;
  constexpr int HEAD_DIM = 64;
  float softmax_scale = 1.0 / sqrt(HEAD_DIM);
  int length = q.size(0);

  flash_attention<Q_BLOCK_SIZE, KV_BLOCK_SIZE, HEAD_DIM>
      <<<length / Q_BLOCK_SIZE, Q_BLOCK_SIZE>>>(
          q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
          o.data_ptr<float>(), length, softmax_scale);
  return o;
}

/// TODO: HED_DIM tiling

/// Flash decoding???
///
/// q: (L, d)
/// k: (L, d)
/// v: (L, d)
/// o: (L, d)
/// number_of_thread_blocks: L / Q_BLOCK_SIZE
///
/// Multiple threads in the thread block with same threadIdx.y will compute
/// a single row in the o. In this case, each thread will do some redudant
/// computation: the o_shared[threadIdx.y] will be computed KV_BLOCK_SIZE times
/// What's more, this method requires multiple synchronization
///
/// blockDim: (Q_BLOCK_SIZE, KV_BLOCK_SIZE)
template <int Q_BLOCK_SIZE, int KV_BLOCK_SIZE, int HEAD_DIM>
__global__ void
flash_attention_kv_parallel(const float *__restrict__ q,
                            const float *__restrict__ k,
                            const float *__restrict__ v, float *__restrict__ o,
                            const int length, float softmax_scale) {
  __shared__ float q_shared[Q_BLOCK_SIZE][HEAD_DIM];
  /// Transpose the k_shared, such that continus threads will access the
  /// continus address to avoid the bank conflict
  __shared__ float k_shared[HEAD_DIM][KV_BLOCK_SIZE];
  __shared__ float v_shared[KV_BLOCK_SIZE][HEAD_DIM];
  __shared__ float o_shared[Q_BLOCK_SIZE][HEAD_DIM];
  __shared__ float qk_shared[Q_BLOCK_SIZE][KV_BLOCK_SIZE];

  constexpr auto NUM_KV_ELEMENTS_PER_THREAD = HEAD_DIM / Q_BLOCK_SIZE;
  constexpr auto NUM_QO_ELEMENTS_PER_THREAD = HEAD_DIM / KV_BLOCK_SIZE;
  const auto row_index = blockIdx.x * Q_BLOCK_SIZE + threadIdx.y;
  const auto q_element_start_ptr = q + row_index * HEAD_DIM;
  const auto o_element_start_ptr = o + row_index * HEAD_DIM;

  float m_old = -INFINITY;
  float m_new = -INFINITY;
  float denominator = 0.0;
  float attentions[KV_BLOCK_SIZE] = {0.0};
  const auto num_iters = length / KV_BLOCK_SIZE;

  /// Load Q_BLOCK into shared memory and set o_shared to 0.0
  /// TODO: vector load
  for (int i = 0; i < NUM_QO_ELEMENTS_PER_THREAD; ++i) {
    int index = i * KV_BLOCK_SIZE + threadIdx.x;
    o_shared[threadIdx.y][index] = 0.0;
    q_shared[threadIdx.y][index] = q_element_start_ptr[index];
  }

  /// Loop over k/v
  for (int kv_iter = 0; kv_iter < num_iters; ++kv_iter) {
    /// Load blocked k/v into shared memory. Load BK * D elements with BQ * BK
    /// threads. Therefore, each thread should load D / BQ elements
    {
      int kv_stride = (kv_iter * KV_BLOCK_SIZE + threadIdx.x) * HEAD_DIM;
      const auto k_element_start_ptr = k + kv_stride;
      const auto v_element_start_ptr = v + kv_stride;
      /// TODO: vector load
#pragma unroll
      for (int load_iter = 0; load_iter < NUM_KV_ELEMENTS_PER_THREAD;
           ++load_iter) {
        int index = load_iter * Q_BLOCK_SIZE + threadIdx.y;
        k_shared[index][threadIdx.x] = k_element_start_ptr[index];
        v_shared[threadIdx.x][index] = v_element_start_ptr[index];
      }

      /// Ensure k/v is loaded. All of the threads in the thread block
      /// will view the correct k/v
      __syncthreads();
    }

    /// Compute q @ k
    float qk_product = 0.0;
    for (int i = 0; i < HEAD_DIM; ++i)
      qk_product += q_shared[threadIdx.y][i] * k_shared[i][threadIdx.x];
    qk_product *= softmax_scale;
    qk_shared[threadIdx.y][threadIdx.x] = qk_product;

    /// Ensure q @ k is computed
    __syncthreads();

    /// Compute row max
    float row_max = -INFINITY;
    for (int i = 0; i < KV_BLOCK_SIZE; ++i) {
      auto qk_i = qk_shared[threadIdx.y][i];
      row_max = row_max > qk_i ? row_max : qk_i;
    }

    m_new = max(row_max, m_new);
    qk_product = __expf(qk_product - m_new);

    /// Compute unscaled attention
    float s = 0.0;
    for (int i = 0; i < KV_BLOCK_SIZE; ++i) {
      auto a = __expf(qk_shared[threadIdx.y][i] - m_new);
      s += a;
      attentions[i] = a;
    }

    /// Compute unscaled denominator
    auto scale_diff = kv_iter == 0 ? 1.0f : __expf(m_old - m_new);
    denominator = scale_diff * denominator + s;

    /// attention @ v, and update o
#pragma unroll
    for (int i = 0; i < NUM_QO_ELEMENTS_PER_THREAD; ++i) {
      auto index = i * KV_BLOCK_SIZE + threadIdx.x;
      float av_product = 0.0;
      for (int j = 0; j < KV_BLOCK_SIZE; ++j)
        av_product += attentions[j] * v_shared[j][index];
      o_shared[threadIdx.y][index] =
          o_shared[threadIdx.y][index] * scale_diff + av_product;
    }

    /// In next iteration, m_new is old now. And we should reset attentions
    m_old = m_new;

    /// Sync threads such that new load will happen after all threads within
    /// thread block have finished the computation
    __syncthreads();
  }

  /// Scale the output with denominator and write to HBM
#pragma unroll
  for (int i = 0; i < NUM_QO_ELEMENTS_PER_THREAD; ++i) {
    int index = i * KV_BLOCK_SIZE + threadIdx.x;
    o_element_start_ptr[index] = o_shared[threadIdx.y][index] / denominator;
  }
}

torch::Tensor attention_kv_parallel(torch::Tensor q, torch::Tensor k,
                                    torch::Tensor v) {
  auto o = torch::zeros_like(q);
  constexpr int Q_BLOCK_SIZE = 32;
  constexpr int KV_BLOCK_SIZE = 32;
  constexpr int HEAD_DIM = 64;
  float softmax_scale = 1.0 / sqrt(HEAD_DIM);
  int length = q.size(0);

  dim3 blockDim(KV_BLOCK_SIZE, Q_BLOCK_SIZE);

  flash_attention_kv_parallel<Q_BLOCK_SIZE, KV_BLOCK_SIZE, HEAD_DIM>
      <<<length / Q_BLOCK_SIZE, blockDim>>>(
          q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
          o.data_ptr<float>(), length, softmax_scale);
  return o;
}
