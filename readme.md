## 说明

本仓库测试 flashinfer 和 flash-attention batch decode 的性能。
起因是观察到 H20 上 flashinfer 要快于 flash-attention 3，相关讨论见：https://github.com/Dao-AILab/flash-attention/issues/1572


目前结果

MHA

```
=== batch_size=16   kv_len=32000 head_dim=128  num_kv_heads=32   num_qo_heads=32   page_size=128 q_dtype=torch.float16 kv_dtype=torch.float16

q_fa.shape=torch.Size([16, 1, 32, 128]) k_cache_paged.shape=torch.Size([4000, 128, 32, 128]) block_table.shape=torch.Size([16, 250])
```

unit: ms

| 硬件           | flash-attention3 | flashinfer | flashinfer (tensor core) |
| -------------- | ---------------- | ---------- | ------------------------ |
| h100(PCIE)     | 4.34             | 4.24       | 4.32                     |
| h20            | 8.27             | 2.46       | 2.48                     |
| a800 80g(SXM) | 6.73             | 4.61       | 4.91                     |
| a800 80g(PCIE) | 8.32             | 5.9        | 5.61                     |


GQA

```
=== batch_size=128  kv_len=16384 head_dim=128  num_kv_heads=16   num_qo_heads=128  page_size=128 q_dtype=torch.float16 kv_dtype=torch.float16

q_fa.shape=torch.Size([128, 1, 128, 128]) k_cache_paged.shape=torch.Size([16384, 128, 16, 128]) block_table.shape=torch.Size([128, 128])
```


| 硬件           | flash-attention3 | flashinfer | flashinfer (tensor core) |
| -------------- | ---------------- | ---------- | ------------------------ |
| h100(PCIE)     | 8.79             | 13.68      | 8.69                     |
| h20            | 16               | 14.68      | 5.05                     |
| a800 80g(SXM) | 13.48            | 15.49      | 9.18                     |
| a800 80g(PCIE) | 16.51            | 19.64      | 11.22                    |
