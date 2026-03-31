import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <vector>

#define NUM_HEADS 16
#define HEAD_DIM_CKV 512
#define HEAD_DIM_KPE 64
#define TOPK 2048
#define LOG2E 1.4426950408889634f

__global__ void dsa_attn_kernel_v2(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ sparse_indices,
    float sm_scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    int num_tokens
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    if (token_idx >= num_tokens) return;

    int wid = tid >> 5;
    int lane = tid & 31;

    const __nv_bfloat16* q_n_ptr = q_nope + ((long long)token_idx * NUM_HEADS + head_idx) * HEAD_DIM_CKV;
    const __nv_bfloat16* q_p_ptr = q_pe + ((long long)token_idx * NUM_HEADS + head_idx) * HEAD_DIM_KPE;

    float4 q_n_0 = ((const float4*)q_n_ptr)[lane];
    float4 q_n_1 = ((const float4*)q_n_ptr)[lane + 32];
    float q_nf[16];
    {
        const __nv_bfloat162* b0 = (const __nv_bfloat162*)&q_n_0;
        const __nv_bfloat162* b1 = (const __nv_bfloat162*)&q_n_1;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 v0 = __bfloat1622float2(b0[i]);
            float2 v1 = __bfloat1622float2(b1[i]);
            q_nf[i * 2] = v0.x;
            q_nf[i * 2 + 1] = v0.y;
            q_nf[8 + i * 2] = v1.x;
            q_nf[8 + i * 2 + 1] = v1.y;
        }
    }
    __nv_bfloat162 q_pe_val = ((const __nv_bfloat162*)q_p_ptr)[lane];
    float2 q_pf = __bfloat1622float2(q_pe_val);

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    const int* idx_base = sparse_indices + (long long)token_idx * TOPK;
    const float scale = sm_scale * LOG2E;
    const int key_start = wid * 256;

    for (int ki = 0; ki < 256; ki += 16) {
        int indices[16];
        unsigned int valid_mask = 0;

        #pragma unroll
        for (int u = 0; u < 16; u++) {
            int idx = idx_base[key_start + ki + u];
            indices[u] = idx;
            if (idx >= 0) valid_mask |= (1u << u);
        }

        if (valid_mask == 0) continue;

        float4 ckv_as[16], ckv_bs[16];
        __nv_bfloat162 kpes[16];

        #pragma unroll
        for (int u = 0; u < 16; u++) {
            if (valid_mask & (1u << u)) {
                long long base = (long long)indices[u];
                ckv_as[u] = ((const float4*)(ckv_cache + base * HEAD_DIM_CKV))[lane];
                ckv_bs[u] = ((const float4*)(ckv_cache + base * HEAD_DIM_CKV))[lane + 32];
                kpes[u] = ((const __nv_bfloat162*)(kpe_cache + base * HEAD_DIM_KPE))[lane];
            }
        }

        #pragma unroll
        for (int u = 0; u < 16; u++) {
            if (!(valid_mask & (1u << u))) continue;

            const __nv_bfloat162* ba = (const __nv_bfloat162*)&ckv_as[u];
            const __nv_bfloat162* bb = (const __nv_bfloat162*)&ckv_bs[u];
            float k_nf[16];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 va = __bfloat1622float2(ba[i]);
                float2 vb = __bfloat1622float2(bb[i]);
                k_nf[i * 2] = va.x;
                k_nf[i * 2 + 1] = va.y;
                k_nf[8 + i * 2] = vb.x;
                k_nf[8 + i * 2 + 1] = vb.y;
            }
            float2 k_pef = __bfloat1622float2(kpes[u]);

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; i++) dot += q_nf[i] * k_nf[i];
            dot += q_pf.x * k_pef.x + q_pf.y * k_pef.y;

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            }

            float s = dot * scale;
            float m_new = fmaxf(m, s);
            float p = exp2f(s - m_new);
            float fac = exp2f(m - m_new);
            m = m_new;
            l = l * fac + p;
            #pragma unroll
            for (int i = 0; i < 16; i++) acc[i] = acc[i] * fac + p * k_nf[i];
        }
    }

    __shared__ float smem_m[8];
    __shared__ float smem_l[8];
    __shared__ float smem_acc[8][512];

    if (lane == 0) {
        smem_m[wid] = m;
        smem_l[wid] = l;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        smem_acc[wid][lane * 8 + i] = acc[i];
        smem_acc[wid][(lane + 32) * 8 + i] = acc[8 + i];
    }

    __syncthreads();

    float m_glob = smem_m[0];
    #pragma unroll
    for (int w = 1; w < 8; w++) m_glob = fmaxf(m_glob, smem_m[w]);

    float l_glob = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; w++) l_glob += smem_l[w] * exp2f(smem_m[w] - m_glob);

    int out_idx = tid * 2;
    float out0 = 0.0f, out1 = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        float fac = exp2f(smem_m[w] - m_glob);
        out0 += smem_acc[w][out_idx] * fac;
        out1 += smem_acc[w][out_idx + 1] * fac;
    }

    float inv_l = (l_glob > 1e-38f) ? (1.0f / l_glob) : 0.0f;
    __nv_bfloat162 out_val = __floats2bfloat162_rn(out0 * inv_l, out1 * inv_l);

    __nv_bfloat16* out_ptr = output + ((long long)token_idx * NUM_HEADS + head_idx) * HEAD_DIM_CKV;
    ((__nv_bfloat162*)out_ptr)[tid] = out_val;

    if (tid == 0) {
        lse[token_idx * NUM_HEADS + head_idx] = (l_glob > 1e-38f) ? (m_glob + log2f(l_glob)) : -FLT_MAX;
    }
}

std::vector<torch::Tensor> dsa_attention_v2(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    double sm_scale
) {
    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be CUDA");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be CUDA");
    TORCH_CHECK(ckv_cache.is_cuda(), "ckv_cache must be CUDA");
    TORCH_CHECK(kpe_cache.is_cuda(), "kpe_cache must be CUDA");
    TORCH_CHECK(sparse_indices.is_cuda(), "sparse_indices must be CUDA");
    TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16, "q_nope must be bfloat16");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "q_pe must be bfloat16");
    TORCH_CHECK(ckv_cache.scalar_type() == torch::kBFloat16, "ckv_cache must be bfloat16");
    TORCH_CHECK(kpe_cache.scalar_type() == torch::kBFloat16, "kpe_cache must be bfloat16");
    TORCH_CHECK(sparse_indices.scalar_type() == torch::kInt32, "sparse_indices must be int32");
    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == NUM_HEADS && q_nope.size(2) == HEAD_DIM_CKV, "q_nope shape mismatch");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(1) == NUM_HEADS && q_pe.size(2) == HEAD_DIM_KPE, "q_pe shape mismatch");
    TORCH_CHECK(ckv_cache.dim() == 3 && ckv_cache.size(1) == 64 && ckv_cache.size(2) == HEAD_DIM_CKV, "ckv_cache shape mismatch");
    TORCH_CHECK(kpe_cache.dim() == 3 && kpe_cache.size(1) == 64 && kpe_cache.size(2) == HEAD_DIM_KPE, "kpe_cache shape mismatch");
    TORCH_CHECK(sparse_indices.dim() == 2 && sparse_indices.size(0) == q_nope.size(0) && sparse_indices.size(1) == TOPK, "sparse_indices shape mismatch");

    auto output = torch::empty({q_nope.size(0), NUM_HEADS, HEAD_DIM_CKV}, q_nope.options());
    auto lse = torch::empty({q_nope.size(0), NUM_HEADS}, q_nope.options().dtype(torch::kFloat32));
    int num_tokens = q_nope.size(0);
    dim3 grid(num_tokens, NUM_HEADS);
    dim3 block(256);
    dsa_attn_kernel_v2<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
        sparse_indices.data_ptr<int>(),
        (float)sm_scale,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        lse.data_ptr<float>(),
        num_tokens
    );
    return {output, lse};
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> dsa_attention_v2(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    double sm_scale
);
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_attention_v31",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["dsa_attention_v2"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _module


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    sparse_indices = sparse_indices.contiguous()
    sm_scale_val = sm_scale.item() if isinstance(sm_scale, torch.Tensor) else float(sm_scale)
    mod = _get_module()
    output, lse = mod.dsa_attention_v2(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_val
    )
    return output, lse
