
import numpy as np
import torch
import math
import pytest
import flashinfer

# from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata

from utils import bench,bench_kineto

def print_error(a, b):
    print(f"MSE: {torch.nn.MSELoss()(a, b)}")
    print(f"MAE: {torch.nn.L1Loss()(a, b)}")
    
def generate_block_kvcache(seqlen_k, page_size, batch_size, nheads_k, head_dim, device, dtype, no_rand=True):
    '''
    @Return
        block_table: (batch_size, seqlen_k / page_size)
    '''
    num_blocks = math.ceil(seqlen_k / page_size) * batch_size
    # kv_data
    k_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, head_dim, device=device
    ).to(dtype)
    v_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, head_dim, device=device
    ).to(dtype)

    nblocks_per_batch = num_blocks // batch_size
    if no_rand:
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(batch_size, nblocks_per_batch)
    else:
        block_table = torch.randperm(num_blocks, dtype=torch.int32, device=device).reshape(batch_size, nblocks_per_batch)

    seqlens_k = torch.tensor([seqlen_k for i in range(batch_size)], dtype=torch.int32, device=device)
    
    # k_scales = torch.empty(nheads_k, device=device, dtype=torch.float32)
    # v_scales = torch.empty(nheads_k, device=device, dtype=torch.float32)
    k_cache_fp8 = torch.empty_like(k_cache_paged, dtype=torch.float8_e4m3fn)
    v_cache_fp8 = torch.empty_like(v_cache_paged, dtype=torch.float8_e4m3fn)

    # for i in range(nheads_k):
    #     k_cache_fp8[:, :, i, :], k_scales[i] = quantize_to_fp8_e4m3(k_cache_paged[:, :, i, :])
    #     v_cache_fp8[:, :, i, :], v_scales[i] = quantize_to_fp8_e4m3(v_cache_paged[:, :, i, :])
    k_cache_fp8, k_scales = quantize_to_fp8_e4m3(k_cache_paged)
    v_cache_fp8, v_scales = quantize_to_fp8_e4m3(v_cache_paged)

    return k_cache_paged, v_cache_paged, block_table, seqlens_k, k_cache_fp8, v_cache_fp8, k_scales, v_scales
def convert_page_table(block_table, seqlens_k, page_size, device):
    '''
    block_table: (batch_size, max_blocks),
    seqlens_k: (batch_size, )
    '''
    batch_size = seqlens_k.shape[0]
    
    kv_indptr = [0]
    for i in range(batch_size):
        kv_indptr.append(kv_indptr[i] + math.ceil(seqlens_k[i].item()/page_size))
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=device)
    
    kv_indices = []
    for i in range(batch_size):
        nblocks = (int(seqlens_k[i])+page_size-1)//page_size
        for j in range(nblocks):
            kv_indices.append(block_table[i][j].item())
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32, device=device)
    
    kv_last_page_len = []
    for i in range(batch_size):
        kv_last_page_len.append((seqlens_k[i].item() - 1 )%page_size + 1)
    kv_last_page_len = torch.tensor(kv_last_page_len, dtype=torch.int32, device=device)

    return kv_indptr, kv_indices, kv_last_page_len

# 辅助函数：将高精度张量转换为 FP8 并计算逆尺度因子
# 该实现参考了 flashinfer.gemm.bmm_fp8 的文档示例 [7]
def quantize_to_fp8_e4m3(tensor: torch.Tensor):
    """
    Quantizes a tensor to float8_e4m3fn and returns the quantized tensor
    and its inverse scale factor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # 计算量化范围
    abs_max = tensor.abs().max()
    # 计算尺度因子，避免除以零
    scale = finfo.max / abs_max.clamp(min=1e-12)
    # 量化并钳位到 FP8 的表示范围
    quantized_tensor = (tensor * scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    # 返回逆尺度因子，用于反量化
    inverse_scale = scale.float().reciprocal()
    return quantized_tensor, inverse_scale

def dump_tensor(weight, weight_name):
    weight = weight.cpu()
    if weight.dtype==torch.bfloat16:
        weight = weight.to(torch.float)
    weight_np = weight.numpy()
    weight_dump = weight_np.ravel()[:1000]

    data_type = weight_dump.dtype
    fmt = '%f'
    if data_type in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
        fmt = '%d'
    elif data_type in [np.float16, np.float32]:
        fmt = '%f'
    
    np.savetxt(f'{weight_name}.txt', weight_dump, fmt)

# @pytest.mark.parametrize("kv_len", [128, 2000, 4000, 32000])
# @pytest.mark.parametrize("batch_size", [16])
# @pytest.mark.parametrize("num_kv_heads", [32])
# @pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("kv_len", [1024, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("num_kv_heads", [16])
@pytest.mark.parametrize("num_qo_heads", [128])
@pytest.mark.parametrize("page_size", [128])
@pytest.mark.parametrize("head_dim", [128])
# @pytest.mark.parametrize("q_dtype", [torch.float16])
# @pytest.mark.parametrize("kv_dtype", [torch.float16])
@pytest.mark.parametrize("q_dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16])
# @pytest.mark.parametrize("q_dtype", [torch.float8_e4m3fn])
# @pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("no_rand", [False])
@pytest.mark.parametrize("use_tensor_cores", [True, False])
@pytest.mark.parametrize("run_bench", [True])
def test_batch_decode_with_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    q_dtype,
    kv_dtype,
    no_rand,  # 生成 pagetable 时，是否打乱
    use_tensor_cores,   # in fact, this option make flashinfer use prefill function
    run_bench
):
    if num_qo_heads < num_kv_heads:
        pytest.skip("num_qo_heads should >= num_kv_heads")
    print(f'\n=== {batch_size=:<4d} {kv_len=:<5d} {head_dim=:<4d} {num_kv_heads=:<4d} {num_qo_heads=:<4d} {page_size=:<4d} {use_tensor_cores=} {q_dtype=} {kv_dtype=}')
    device = "cuda:0"
    torch.manual_seed(42)
    
    try:
        q = torch.randn(batch_size, num_qo_heads, head_dim, device=device).to(q_dtype)
        q_fp8, q_scales = quantize_to_fp8_e4m3(q)
        q_fa = torch.unsqueeze(q, dim=1)
        
        num_pages_per_seq = (kv_len + page_size - 1) // page_size
        total_num_pages = num_pages_per_seq * batch_size
        
        # flash-attention
        k_cache_paged, v_cache_paged, block_table, seqlens_k, \
        k_cache_fp8, v_cache_fp8, k_scales, v_scales = generate_block_kvcache(
            kv_len, page_size, batch_size, num_kv_heads, head_dim, device, kv_dtype, no_rand
        )

        # flashinfer
        kv_data = torch.stack((k_cache_paged, v_cache_paged), dim=1)
        kv_data_fp8 = torch.stack((k_cache_fp8, v_cache_fp8), dim=1)
    
        # kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]  # NHD
        # kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        # kv_data = kv_data_fp32.to(kv_dtype)
        kv_indptr, kv_indices, kv_last_page_len = convert_page_table(block_table, seqlens_k, page_size, device)
        # # 这里构造了一个最简单情况，所有请求有 kv_len/page_size 个 page，page 索引按序递增。
        # kv_indptr = (
        #     torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        #     * num_pages_per_seq
        # )
        # kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
        # kv_last_page_len = torch.full(
        #     (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
        # )
    except torch.OutOfMemoryError:
        pytest.skip("OOM")
    
    # scheduler_metadata = get_scheduler_metadata(
    #     batch_size, 1, kv_len, num_qo_heads, num_kv_heads, head_dim,
    #     seqlens_k, q_dtype, headdim_v=head_dim, page_size=page_size, causal=True
    # )
    print(f"{q_fa.shape=} {k_cache_paged.shape=} {block_table.shape=}")
    def run_fa3():
        o_fa_ = flash_attn_with_kvcache(
            q_fa,
            k_cache_paged,
            v_cache_paged,
            k=None,
            v=None,
            rotary_cos=None,
            rotary_sin=None,
            cache_seqlens=seqlens_k,
            cache_batch_idx=None,
            cache_leftpad=None,
            # block_table=block_table,
            page_table=block_table, # fa3 change to page_table
            # causal=causal,
            causal=True,
            # window_size=window_size,
            # rotary_interleaved=rotary_interleaved,
            # alibi_slopes=alibi_slopes,
            # num_splits=num_splits,
            scheduler_metadata=scheduler_metadata
        )
        o_fa = torch.squeeze(o_fa_, dim=1)
        return o_fa
    # o_fa = run_fa3()
    
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout='NHD', use_tensor_cores=use_tensor_cores)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        logits_soft_cap=0.0,
        pos_encoding_mode='NONE',  # ROPE_LLAMA
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    
    wrapper_fp8 = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout='NHD', use_tensor_cores=use_tensor_cores)
    wrapper_fp8.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        logits_soft_cap=0.0,
        pos_encoding_mode='NONE',  # ROPE_LLAMA
        q_data_type=q_dtype,
        # q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
    )
    def run_flashinfer():
        o = wrapper.run(q, kv_data, return_lse=False)
        return o
    def run_flashinfer_fp8():
        # o = wrapper_fp8.run(q_fp8, kv_data_fp8, q_scale=q_scales, k_scale=k_scales, v_scale=v_scales, return_lse=False)
        o = wrapper_fp8.run(q, kv_data_fp8, k_scale=k_scales, v_scale=v_scales, return_lse=False)
        return o
    o = run_flashinfer()
    o2 = run_flashinfer_fp8()
    # print(o.dtype)
    # print(o2.dtype)

    # dump_tensor(o, 'flashinfer_o')
    # dump_tensor(o2, 'flashinfer_fp8_kv_o')

    #测试精度
    
    # print(f"{o.dtype=}, {o.shape=}")
    # print(f"{o_fa.dtype=}, {o_fa.shape=}")
    # if q_dtype in [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
    #     print(f"Convert q:{q_dtype} to float32 for comparison using torch")
    #     o = o.to(torch.float32)
    #     o_fa = o_fa.to(torch.float32)
    
    # np.savetxt('o_fa.txt', o_fa.cpu().flatten().numpy(), fmt='%.6f')
    # np.savetxt('o.txt', o.cpu().flatten().numpy(), fmt='%.6f')
    
    # print_error(o, o_fa)
    # torch.testing.assert_close(o.cpu(), o_fa.cpu(), rtol=1e-3, atol=1e-3)
    
    if run_bench:
        # t1 = bench(run_fa3)
        t2 = bench(run_flashinfer)
        t3 = bench(run_flashinfer_fp8)
        # print(f'{"flash-attn-3":20}: {t1}ms')
        print(f'{"flashinfer":20}: {t2:.3f}ms')
        print(f'{"flashinfer_fp8":20}: {t3:.3f}ms   x{t3/t2:.2f}')
    
if __name__=="__main__":
    test_batch_decode_with_paged_kv_cache(
        batch_size=16,
        kv_len=32000,
        page_size=8,
        num_kv_heads=4,
        num_qo_heads=32,
        head_dim=128,
        # q_dtype=torch.float8_e4m3fn,
        # kv_dtype=torch.float8_e4m3fn,
        # q_dtype=torch.float8_e5m2,
        # kv_dtype=torch.float8_e5m2,
        q_dtype=torch.bfloat16,
        # kv_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.bfloat16,
        no_rand=False,
        use_tensor_cores=False,
        run_bench=True
    )