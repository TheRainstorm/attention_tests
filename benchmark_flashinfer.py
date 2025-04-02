
import numpy as np
import torch
import math
import pytest
import flashinfer

import flash_attn_interface  # fa3
flash_attn_with_kvcache = flash_attn_interface.flash_attn_with_kvcache

def generate_block_kvcache(seqlen_k, page_size, batch_size, nheads_k, head_dim, device, dtype, no_rand=True):
    '''
    @Return
        block_table: (batch_size, seqlen_k / page_size)
        
    '''
    num_blocks = math.ceil(seqlen_k / page_size) * batch_size
    # kv_data
    k_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, head_dim, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, head_dim, device=device, dtype=dtype
    )

    nblocks_per_batch = num_blocks // batch_size
    if no_rand:
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(batch_size, nblocks_per_batch)
    else:
        block_table = torch.randperm(num_blocks, dtype=torch.int32, device=device).reshape(batch_size, nblocks_per_batch)

    seqlens_k = torch.tensor([seqlen_k for i in range(batch_size)], dtype=torch.int32, device=device)
    
    return k_cache_paged, v_cache_paged, block_table, seqlens_k

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

@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048, 16384])
# @pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("page_size", [1256])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
# @pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
# @pytest.mark.parametrize("logits_soft_cap", [0.0])
# @pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
# @pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_decode_with_paged_kv_cache(
    batch_size=12,
    kv_len=2048,
    page_size=256,
    num_kv_heads=4,
    num_qo_heads=32,
    head_dim=128,
    kv_layout='NHD',
    pos_encoding_mode='NONE',
    return_lse=False,
    logits_soft_cap=0.0,
    q_dtype=torch.float16,
    kv_dtype=torch.float16,
    contiguous_kv=True,
):
    device = "cuda:0"
    q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=q_dtype)
    q_fa = torch.unsqueeze(q, dim=1)
    
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    
    # flash-attention
    k_cache_paged, v_cache_paged, block_table, seqlens_k = generate_block_kvcache(
        kv_len, page_size, batch_size, num_kv_heads, head_dim, device, kv_dtype
    )
    
    # flashinfer
    kv_data = torch.stack((k_cache_paged, v_cache_paged), dim=1)
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
        # window_size=window_size,
        # rotary_interleaved=rotary_interleaved,
        # alibi_slopes=alibi_slopes,
        # num_splits=num_splits,
    )
    o_fa = torch.squeeze(o_fa_, dim=1)
    
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        logits_soft_cap=logits_soft_cap,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o = wrapper.run(q, kv_data)
    
    # np.savetxt('o_fa.txt', o_fa.cpu().flatten().numpy(), fmt='%.6f')
    # np.savetxt('o.txt', o.cpu().flatten().numpy(), fmt='%.6f')
    
    def print_error(a, b):
        print(f"MSE: {torch.nn.MSELoss()(a, b)}")
        print(f"MAE: {torch.nn.L1Loss()(a, b)}")
    print_error(o, o_fa)
    torch.testing.assert_close(o.cpu(), o_fa.cpu(), rtol=1e-3, atol=1e-3)

if __name__=="__main__":
    test_batch_decode_with_paged_kv_cache(
        batch_size=12,
        # kv_len=54,
        kv_len=2048,
        # page_size=1,
        page_size=256,
        num_kv_heads=4,
        num_qo_heads=32,
        head_dim=128,
        q_dtype=torch.float16,
        kv_dtype=torch.float16
    )