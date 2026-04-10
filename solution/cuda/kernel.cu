// submission-v2
// CUDA sparse-attention implementation for
// dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.
// Exported torch binding symbol: dsa_forward.

#include <torch/extension.h>
#include <tuple>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared_sum[4];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared_sum[wid] = val;
    __syncthreads();

    float sum = 0.0f;
    if (threadIdx.x == 0) {
        sum = shared_sum[0] + shared_sum[1] + shared_sum[2] + shared_sum[3];
        shared_sum[0] = sum;
    }
    __syncthreads();
    
    return shared_sum[0];
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, tmp);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared_max[4];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) shared_max[wid] = val;
    __syncthreads();

    float m = -INFINITY;
    if (threadIdx.x == 0) {
        m = shared_max[0];
        m = fmaxf(m, shared_max[1]);
        m = fmaxf(m, shared_max[2]);
        m = fmaxf(m, shared_max[3]);
        shared_max[0] = m;
    }
    __syncthreads();
    
    return shared_max[0];
}

__global__ void dsa_forward_splitk_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t* __restrict__ sparse_indices,
    float* __restrict__ mid_out,
    float* __restrict__ mid_m,
    float* __restrict__ mid_d,
    int topk,
    float sm_scale,
    int chunk_size
) {
    const int chunk_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int token_idx = blockIdx.z;
    const int tid       = threadIdx.x;

    int start_k = chunk_idx * chunk_size;
    int end_k   = min(start_k + chunk_size, topk);

    const int qo_heads = 16;
    const __nv_bfloat16* q_nope_ptr = q_nope + (token_idx * qo_heads + head_idx) * 512;
    const __nv_bfloat16* q_pe_ptr   = q_pe   + (token_idx * qo_heads + head_idx) * 64;
    const int32_t* sparse_idx_ptr   = sparse_indices + token_idx * topk;

    // Load query parts into registers using float4 (8 bf16 values per thread)
    nv_bfloat162 q_regs[4];
    #pragma unroll
    for(int i = 0; i < 4; ++i) q_regs[i] = __float22bfloat162_rn(make_float2(0.0f, 0.0f));

    if (tid < 64) {
        float4 q_load = reinterpret_cast<const float4*>(q_nope_ptr)[tid];
        memcpy(q_regs, &q_load, 16);
    } else if (tid < 72) {
        float4 q_load = reinterpret_cast<const float4*>(q_pe_ptr)[tid - 64];
        memcpy(q_regs, &q_load, 16);
    }

    float m_max = -INFINITY;
    float d_sum = 0.0f;
    float out_acc[8] = {0.0f};

    // Iterate over the Top-K KV elements assigned to this chunk
    for (int k = start_k; k < end_k; ++k) {
        int sparse_idx = sparse_idx_ptr[k];
        
        // Safety constraint: mask out padded indices
        if (sparse_idx == -1) continue;

        nv_bfloat162 k_regs[4];
        #pragma unroll
        for(int i = 0; i < 4; ++i) k_regs[i] = __float22bfloat162_rn(make_float2(0.0f, 0.0f));

        if (tid < 64) {
            float4 k_load = reinterpret_cast<const float4*>(ckv_cache + sparse_idx * 512)[tid];
            memcpy(k_regs, &k_load, 16);
        } else if (tid < 72) {
            float4 k_load = reinterpret_cast<const float4*>(kpe_cache + sparse_idx * 64)[tid - 64];
            memcpy(k_regs, &k_load, 16);
        }

        // Thread-level dot product execution in FP32
        float sum = 0.0f;
        if (tid < 72) {
            #pragma unroll
            for(int i = 0; i < 4; ++i) {
                float2 q_f2 = __bfloat1622float2(q_regs[i]);
                float2 k_f2 = __bfloat1622float2(k_regs[i]);
                sum = fmaf(q_f2.x, k_f2.x, sum);
                sum = fmaf(q_f2.y, k_f2.y, sum);
            }
        }

        // Block reduction
        float logit = block_reduce_sum(sum);
        // Scale by log2(e) to perform all Softmax math in base-2 directly
        logit *= sm_scale * 1.4426950408889634f;

        // Online Softmax (Base-2)
        float m_max_new = fmaxf(m_max, logit);
        float weight    = exp2f(logit - m_max_new);
        float scale_old = exp2f(m_max - m_max_new);

        d_sum = d_sum * scale_old + weight;
        m_max = m_max_new;

        // Accumulate output directly in registers
        if (tid < 64) {
            #pragma unroll
            for(int i = 0; i < 4; ++i) {
                float2 k_f2 = __bfloat1622float2(k_regs[i]);
                out_acc[i*2]   = out_acc[i*2]   * scale_old + weight * k_f2.x;
                out_acc[i*2+1] = out_acc[i*2+1] * scale_old + weight * k_f2.y;
            }
        }
    }

    // Write intermediate states to global memory for Stage 2 reduction
    if (tid == 0) {
        int idx = (token_idx * 16 + head_idx) * gridDim.x + chunk_idx;
        mid_m[idx] = m_max;
        mid_d[idx] = d_sum;
    }
    
    __shared__ float smem_out[512];
    if (tid < 64) {
        reinterpret_cast<float4*>(smem_out)[tid * 2 + 0] = make_float4(out_acc[0], out_acc[1], out_acc[2], out_acc[3]);
        reinterpret_cast<float4*>(smem_out)[tid * 2 + 1] = make_float4(out_acc[4], out_acc[5], out_acc[6], out_acc[7]);
    }
    __syncthreads();

    // Now 128 threads write 128 contiguous float4 to mid_out
    int base_vec_idx = ((token_idx * 16 + head_idx) * gridDim.x + chunk_idx) * 128;
    float4 out_val = reinterpret_cast<float4*>(smem_out)[tid];
    reinterpret_cast<float4*>(mid_out)[base_vec_idx + tid] = out_val;
}

__global__ void dsa_forward_reduce_kernel(
    const float* __restrict__ mid_out,
    const float* __restrict__ mid_m,
    const float* __restrict__ mid_d,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    int num_chunks
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x; // 128 threads
    
    int chunk_base = (token_idx * 16 + head_idx) * num_chunks;
    
    // Step 1: Find global maximum of logit_scaled
    float local_m_max = -INFINITY;
    for (int c = tid; c < num_chunks; c += blockDim.x) {
        float m = mid_m[chunk_base + c];
        local_m_max = fmaxf(local_m_max, m);
    }
    float global_m_max = block_reduce_max(local_m_max);
    
    // Edge case: All padding indices evaluated
    if (global_m_max == -INFINITY) {
        nv_bfloat162 zero_bf = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
        nv_bfloat162 zeros[2] = {zero_bf, zero_bf};
        float2 zero;
        memcpy(&zero, zeros, 8);
        reinterpret_cast<float2*>(output)[(token_idx * 16 + head_idx) * 128 + tid] = zero;
        
        if (tid == 0) lse[token_idx * 16 + head_idx] = -INFINITY;
        return;
    }

    // Step 2: Compute global denominator sum
    float local_d_sum = 0.0f;
    for (int c = tid; c < num_chunks; c += blockDim.x) {
        float m = mid_m[chunk_base + c];
        float d = mid_d[chunk_base + c];
        local_d_sum += d * exp2f(m - global_m_max);
    }
    float global_d_sum = block_reduce_sum(local_d_sum);

    // Step 3: Compute final normalized attention outputs
    float final_vals[4] = {0.0f};
    int out_vec_base = (token_idx * 16 + head_idx) * num_chunks * 128; // in float4
    
    for (int c = 0; c < num_chunks; ++c) {
        float m = mid_m[chunk_base + c];
        float weight = exp2f(m - global_m_max);
        
        float4 out_val = reinterpret_cast<const float4*>(mid_out)[out_vec_base + c * 128 + tid];
        
        final_vals[0] += out_val.x * weight;
        final_vals[1] += out_val.y * weight;
        final_vals[2] += out_val.z * weight;
        final_vals[3] += out_val.w * weight;
    }
    
    float inv_sum = global_d_sum > 0.0f ? 1.0f / global_d_sum : 0.0f;
    nv_bfloat162 out_bf[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float2 f2 = make_float2(final_vals[i*2] * inv_sum, final_vals[i*2+1] * inv_sum);
        out_bf[i] = __float22bfloat162_rn(f2);
    }
    
    float2 out_write;
    memcpy(&out_write, out_bf, 8);
    reinterpret_cast<float2*>(output)[(token_idx * 16 + head_idx) * 128 + tid] = out_write;
    
    // Step 4: Write Base-2 Log-Sum-Exp
    if (tid == 0) {
        if (global_d_sum > 0.0f) {
            // Because global_m_max is ALREADY scaled to base-2, we just add log2f(d_sum)
            lse[token_idx * 16 + head_idx] = global_m_max + log2f(global_d_sum);
        } else {
            lse[token_idx * 16 + head_idx] = -INFINITY;
        }
    }
}

// C++ Entry Point Mapping
std::tuple<torch::Tensor, torch::Tensor> dsa_forward(
    torch::Tensor q_nope,         // [num_tokens, 16, 512]  bfloat16
    torch::Tensor q_pe,           // [num_tokens, 16, 64]   bfloat16
    torch::Tensor ckv_cache,      // [num_pages, 64, 512]   bfloat16
    torch::Tensor kpe_cache,      // [num_pages, 64, 64]    bfloat16
    torch::Tensor sparse_indices, // [num_tokens, 2048]     int32
    float sm_scale                // scalar: 1/sqrt(192)
) {
    TORCH_CHECK(q_nope.is_contiguous(), "q_nope must be contiguous");
    TORCH_CHECK(q_pe.is_contiguous(), "q_pe must be contiguous");
    TORCH_CHECK(ckv_cache.is_contiguous(), "ckv_cache must be contiguous");
    TORCH_CHECK(kpe_cache.is_contiguous(), "kpe_cache must be contiguous");
    TORCH_CHECK(sparse_indices.is_contiguous(), "sparse_indices must be contiguous");

    int num_tokens = q_nope.size(0);
    int topk = sparse_indices.size(1);

    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q_nope.device());
    auto options_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device());

    torch::Tensor output = torch::empty({num_tokens, 16, 512}, options_bf16);
    torch::Tensor lse = torch::empty({num_tokens, 16}, options_f32);

    if (num_tokens > 0 && topk > 0) {
        int chunk_size = 64;
        int num_chunks = (topk + chunk_size - 1) / chunk_size;

        // Allocate FP32 intermediates for the two-stage Split-K reduction
        torch::Tensor mid_out = torch::empty({num_tokens, 16, num_chunks, 512}, options_f32);
        torch::Tensor mid_m   = torch::empty({num_tokens, 16, num_chunks}, options_f32);
        torch::Tensor mid_d   = torch::empty({num_tokens, 16, num_chunks}, options_f32);

        // Stage 1: Split-K compute grid. Yields ~512 blocks for large parallelization.
        dim3 grid1(num_chunks, 16, num_tokens);
        dim3 block1(128); 
        
        dsa_forward_splitk_kernel<<<grid1, block1>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            sparse_indices.data_ptr<int32_t>(),
            mid_out.data_ptr<float>(),
            mid_m.data_ptr<float>(),
            mid_d.data_ptr<float>(),
            topk,
            sm_scale,
            chunk_size
        );

        // Stage 2: Final global reduction.
        dim3 grid2(num_tokens, 16);
        dim3 block2(128);
        
        dsa_forward_reduce_kernel<<<grid2, block2>>>(
            mid_out.data_ptr<float>(),
            mid_m.data_ptr<float>(),
            mid_d.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_chunks
        );
    } else if (num_tokens > 0 && topk == 0) {
        output.fill_(0.0f);
        lse.fill_(-INFINITY);
    }

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_forward", &dsa_forward, "DSA sparse attention forward kernel Split-K");
}
