1. Tensor 创建方法

| 函数 | 功能 | 示例 |
| :--- | :--- | :--- |
| `torch.Tensor(*sizes)` | 基础构造函数（未初始化） | `t.Tensor(2, 3)` |
| `torch.tensor(data)` | 从数据创建（推荐，类型安全） | `t.tensor([[1,2],[3,4]])` |
| `torch.ones(*sizes)` | 创建全1张量 | `t.ones(2, 3)` |
| `torch.zeros(*sizes)` | 创建全0张量 | `t.zeros(2, 3)` |
| `torch.eye(n, m)` | 创建单位矩阵 | `t.eye(2, 3)` |
| `torch.arange(start, end, step)` | 创建等差序列 | `t.arange(1, 6, 2)` |
| `torch.linspace(start, end, steps)` | 创建线性间隔序列 | `t.linspace(1, 10, 3)` |
| `torch.rand(*sizes)` | 创建[0,1)均匀分布随机张量 | `t.rand(2, 3)` |
| `torch.randn(*sizes)` | 创建标准正态分布随机张量 | `t.randn(2, 3)` |
| `torch.randperm(n)` | 创建0到n-1的随机排列 | `t.randperm(5)` |
| `tensor.new_*` / `torch.*_like` | 创建与输入同类型/形状的新张量 | `x.new_ones(2,3)`, `torch.zeros_like(x)` |

2. Tensor 基本操作

| 操作类型 | 描述 | 示例 |
| :--- | :--- | :--- |
| 逐元素操作 | 对每个元素独立操作，输出形状不变 | `t.cos(a)`, `a % 3`, `t.clamp(a, min, max)` |
| 归并操作 | 沿指定维度聚合，输出形状可能变小 | `a.sum(dim=0)`, `a.mean()`, `a.cumsum(dim=1)` |
| 比较操作 | 逐元素或整体比较 | `a > b`, `t.max(a)`, `t.max(a, dim=1)`, `t.max(a, b)` |
| 线性代数 | 矩阵运算 | `a.mm(b)`, `a.t()`, `a.inverse()` |
| 索引操作 | 选取特定元素 | `a[0]`, `a[:,1]`, `a[a>0]`, `a.gather(dim, index)` |
| 高级索引 | 使用列表/张量进行复杂索引 | `x[[1,0], [1,1], [2,0]]` |

常见的逐元素操作

|函数|功能|
|:--:|:--:|
|abs/sqrt/div/exp/fmod/log/pow..|绝对值/平方根/除法/指数/求余/对数/求幂..|
|cos/sin/asin/atan2/cosh..|三角函数|
|ceil/round/floor/trunc| 上取整/四舍五入/下取整/只保留整数部分|
|clamp(input,min,max)|超过min和max部分截断|
|sigmod/tanh..|激活函数

常用归并操作

|函数|功能|
|:---:|:---:|
|mean/sum/median/mode|均值/求和/中位数/众数|
|norm/dist|范数/距离|
|std/var|标准差/方差|
|cumsum/cumprod|累加/累乘|

常用比较函数

|函数|功能|
|:--:|:--:|
|gt/lt/ge/le/eq/ne|大于/小于/大于等于/小于等于/等于/不等于|
|topk|最大的k个数|
|sort|排序|
|max/min|比较两个Tensor的最大、最小值|

常用的线性代数函数
|函数|功能|
|:---:|:---:|
|trace|对角线元素之和(矩阵的迹)|
|diag|对角线元素|
|triu/tril|矩阵的上三角/下三角，可指定偏移量|
|mm/bmm|矩阵乘法，batch的矩阵乘法|
|addmm/addbmm/addmv|矩阵运算
|t|转置|
|dot/cross|内积/外积
|inverse|求逆矩阵
|svd|奇异值分解

3. Tensor 形状与维度操作

| 操作 | 功能 | 备注 |
| :--- | :--- | :--- |
| `.view(*shape)` | 改变张量形状（要求内存连续） | `x.view(-1, 8)` |
| `.reshape(*shape)` | 改变张量形状（更安全，可处理不连续内存） | `x.reshape(-1, 8)` |
| `.squeeze()` | 移除所有大小为1的维度 | `(1,3,1,4) -> (3,4)` |
| `.unsqueeze(dim)` | 在指定位置增加一个大小为1的维度 | `x.unsqueeze(0)` |
| `.flatten(start_dim, end_dim)` | 将连续维度展平 | `(2,3,4,5) -> (2,12,5)` |
| `.transpose(dim0, dim1)` | 交换两个维度 | `x.transpose(0, 2)` |
| `.permute(*dims)` | 任意重排维度顺序 | `x.permute(2, 1, 0)` |

4. Tensor 属性与转换

| 属性/方法 | 功能 |
| :--- | :--- |
| `.size()` / `.shape` | 返回张量形状 (`torch.Size`) |
| `.dim()` | 返回张量维度数 |
| `.numel()` | 返回张量中元素总数 |
| `.dtype` | 返回数据类型 |
| `.device` | 返回所在设备 (CPU/GPU) |
| `.item()` | 将单元素张量转为Python标量 |
| `.tolist()` | 将张量转为Python列表 |
| `.clone()` | 创建数据和梯度历史的深拷贝 |
| `.detach()` | 分离张量，不追踪梯度（共享数据） |
| `.to(device/dtype)` | 移动到指定设备或转换数据类型 |

5. 自动微分 (Autograd)

| 概念 | 描述 |
| :--- | :--- |
| `requires_grad` | 标记张量是否需要计算梯度。`True`时，其上的操作会被记录。 |
| `is_leaf` | 判断张量是否为计算图的叶子节点。用户直接创建且`requires_grad=True`的张量是叶子节点。 |
| `.grad` | 存储该张量的梯度。仅当`is_leaf=True`且`requires_grad=True`时，梯度会被保留。 |
| `.backward()` | 从当前张量开始反向传播，计算梯度。 |
| `.grad_fn` | 指向创建该张量的函数（非叶子节点才有）。 |

6. Tensor 与 NumPy 互操作

| 操作 | 共享内存? | 描述 |
| :--- | :--- | :--- |
| `torch.from_numpy(ndarray)` | 是 | 从NumPy数组创建Tensor，共享内存（需数据类型匹配）。 |
| `torch.tensor(ndarray)` | 否 | 从NumPy数组创建Tensor，总是进行数据拷贝。 |
| `tensor.numpy()` | 是 | 将Tensor转为NumPy数组（仅限CPU Tensor），共享内存。 |

