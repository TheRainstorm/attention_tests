import torch
import time

# 配置参数
device = 'cuda'
tensor_size = (10000,)  # 10,000维张量
warmup_iters = 100      # 预热迭代次数
test_iters = 1000       # 正式测试迭代次数

# 创建CUDA张量（禁用梯度跟踪）
x_loop = torch.zeros(tensor_size, device=device)
x_vector = torch.zeros(tensor_size, device=device)

# 预热GPU（避免冷启动误差）
with torch.no_grad():
    for _ in range(warmup_iters):
        x_loop[:] = 2.0
        x_vector[:] = 2.0
torch.cuda.synchronize()

# 测试逐元素修改
start_time = time.time()
with torch.no_grad():
    for i in range(x_loop.shape[0]):
        x_loop[i] = x_loop[i] * 0.5  # 每次修改一个元素
torch.cuda.synchronize()
loop_time = time.time() - start_time

# 测试批量操作
start_time = time.time()
with torch.no_grad():
    x_vector = x_vector * 0.5  # 单次批量赋值
torch.cuda.synchronize()
vector_time = time.time() - start_time

# 输出结果
print(f"逐元素修改耗时: {loop_time:.4f}s")
print(f"批量操作耗时: {vector_time:.4f}s")
print(f"速度差异倍数: {loop_time/vector_time:.1f}x")