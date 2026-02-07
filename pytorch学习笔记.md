{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d9f9d67b-fbf2-490c-a252-a8071d872443",
   "metadata": {},
   "source": [
    "1. Tensor 创建方法"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d40e4164-e823-4c08-9794-e6add3712a29",
   "metadata": {
    "scrolled": true
   },
   "source": [
    "| 函数 | 功能 | 示例 |\n",
    "| :--- | :--- | :--- |\n",
    "| `torch.Tensor(*sizes)` | 基础构造函数（未初始化） | `t.Tensor(2, 3)` |\n",
    "| `torch.tensor(data)` | 从数据创建（推荐，类型安全） | `t.tensor([[1,2],[3,4]])` |\n",
    "| `torch.ones(*sizes)` | 创建全1张量 | `t.ones(2, 3)` |\n",
    "| `torch.zeros(*sizes)` | 创建全0张量 | `t.zeros(2, 3)` |\n",
    "| `torch.eye(n, m)` | 创建单位矩阵 | `t.eye(2, 3)` |\n",
    "| `torch.arange(start, end, step)` | 创建等差序列 | `t.arange(1, 6, 2)` |\n",
    "| `torch.linspace(start, end, steps)` | 创建线性间隔序列 | `t.linspace(1, 10, 3)` |\n",
    "| `torch.rand(*sizes)` | 创建[0,1)均匀分布随机张量 | `t.rand(2, 3)` |\n",
    "| `torch.randn(*sizes)` | 创建标准正态分布随机张量 | `t.randn(2, 3)` |\n",
    "| `torch.randperm(n)` | 创建0到n-1的随机排列 | `t.randperm(5)` |\n",
    "| `tensor.new_*` / `torch.*_like` | 创建与输入同类型/形状的新张量 | `x.new_ones(2,3)`, `torch.zeros_like(x)` |\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fb64ec6b-df70-4de6-b896-484aea6903d1",
   "metadata": {},
   "source": [
    "2. Tensor 基本操作"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5fb84e3-a622-4e1d-8f69-cff2431d17f5",
   "metadata": {},
   "source": [
    "| 操作类型 | 描述 | 示例 |\n",
    "| :--- | :--- | :--- |\n",
    "| 逐元素操作 | 对每个元素独立操作，输出形状不变 | `t.cos(a)`, `a % 3`, `t.clamp(a, min, max)` |\n",
    "| 归并操作 | 沿指定维度聚合，输出形状可能变小 | `a.sum(dim=0)`, `a.mean()`, `a.cumsum(dim=1)` |\n",
    "| 比较操作 | 逐元素或整体比较 | `a > b`, `t.max(a)`, `t.max(a, dim=1)`, `t.max(a, b)` |\n",
    "| 线性代数 | 矩阵运算 | `a.mm(b)`, `a.t()`, `a.inverse()` |\n",
    "| 索引操作 | 选取特定元素 | `a[0]`, `a[:,1]`, `a[a>0]`, `a.gather(dim, index)` |\n",
    "| 高级索引 | 使用列表/张量进行复杂索引 | `x[[1,0], [1,1], [2,0]]` |\n",
    "\n",
    "常见的逐元素操作\n",
    "\n",
    "|函数|功能|\n",
    "|:--:|:--:|\n",
    "|abs/sqrt/div/exp/fmod/log/pow..|绝对值/平方根/除法/指数/求余/对数/求幂..|\n",
    "|cos/sin/asin/atan2/cosh..|三角函数|\n",
    "|ceil/round/floor/trunc| 上取整/四舍五入/下取整/只保留整数部分|\n",
    "|clamp(input,min,max)|超过min和max部分截断|\n",
    "|sigmod/tanh..|激活函数\n",
    "\n",
    "常用归并操作\n",
    "\n",
    "|函数|功能|\n",
    "|:---:|:---:|\n",
    "|mean/sum/median/mode|均值/求和/中位数/众数|\n",
    "|norm/dist|范数/距离|\n",
    "|std/var|标准差/方差|\n",
    "|cumsum/cumprod|累加/累乘|\n",
    "\n",
    "常用比较函数\n",
    "\n",
    "|函数|功能|\n",
    "|:--:|:--:|\n",
    "|gt/lt/ge/le/eq/ne|大于/小于/大于等于/小于等于/等于/不等于|\n",
    "|topk|最大的k个数|\n",
    "|sort|排序|\n",
    "|max/min|比较两个Tensor的最大、最小值|\n",
    "\n",
    "常用的线性代数函数\n",
    "|函数|功能|\n",
    "|:---:|:---:|\n",
    "|trace|对角线元素之和(矩阵的迹)|\n",
    "|diag|对角线元素|\n",
    "|triu/tril|矩阵的上三角/下三角，可指定偏移量|\n",
    "|mm/bmm|矩阵乘法，batch的矩阵乘法|\n",
    "|addmm/addbmm/addmv|矩阵运算\n",
    "|t|转置|\n",
    "|dot/cross|内积/外积\n",
    "|inverse|求逆矩阵\n",
    "|svd|奇异值分解"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04d52f48-0d14-40ec-93aa-de76958d370d",
   "metadata": {},
   "source": [
    "3. Tensor 形状与维度操作"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "336ba0c0-7c23-4c1a-9212-9dfb4ecd6c8d",
   "metadata": {},
   "source": [
    "| 操作 | 功能 | 备注 |\n",
    "| :--- | :--- | :--- |\n",
    "| `.view(*shape)` | 改变张量形状（要求内存连续） | `x.view(-1, 8)` |\n",
    "| `.reshape(*shape)` | 改变张量形状（更安全，可处理不连续内存） | `x.reshape(-1, 8)` |\n",
    "| `.squeeze()` | 移除所有大小为1的维度 | `(1,3,1,4) -> (3,4)` |\n",
    "| `.unsqueeze(dim)` | 在指定位置增加一个大小为1的维度 | `x.unsqueeze(0)` |\n",
    "| `.flatten(start_dim, end_dim)` | 将连续维度展平 | `(2,3,4,5) -> (2,12,5)` |\n",
    "| `.transpose(dim0, dim1)` | 交换两个维度 | `x.transpose(0, 2)` |\n",
    "| `.permute(*dims)` | 任意重排维度顺序 | `x.permute(2, 1, 0)` |\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8dd6ca3-d3f0-4700-80ee-c047fc1f6e8e",
   "metadata": {},
   "source": [
    "4. Tensor 属性与转换"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05fc4128-1dbe-434c-870d-a6f5afe0dd80",
   "metadata": {},
   "source": [
    "| 属性/方法 | 功能 |\n",
    "| :--- | :--- |\n",
    "| `.size()` / `.shape` | 返回张量形状 (`torch.Size`) |\n",
    "| `.dim()` | 返回张量维度数 |\n",
    "| `.numel()` | 返回张量中元素总数 |\n",
    "| `.dtype` | 返回数据类型 |\n",
    "| `.device` | 返回所在设备 (CPU/GPU) |\n",
    "| `.item()` | 将单元素张量转为Python标量 |\n",
    "| `.tolist()` | 将张量转为Python列表 |\n",
    "| `.clone()` | 创建数据和梯度历史的深拷贝 |\n",
    "| `.detach()` | 分离张量，不追踪梯度（共享数据） |\n",
    "| `.to(device/dtype)` | 移动到指定设备或转换数据类型 |\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b685bec5-d9ed-46f3-a57b-62a37f14b96f",
   "metadata": {},
   "source": [
    "5. 自动微分 (Autograd)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f4407643-9f7d-4979-b92a-1f316e70dda8",
   "metadata": {},
   "source": [
    "| 概念 | 描述 |\n",
    "| :--- | :--- |\n",
    "| `requires_grad` | 标记张量是否需要计算梯度。`True`时，其上的操作会被记录。 |\n",
    "| `is_leaf` | 判断张量是否为计算图的叶子节点。用户直接创建且`requires_grad=True`的张量是叶子节点。 |\n",
    "| `.grad` | 存储该张量的梯度。仅当`is_leaf=True`且`requires_grad=True`时，梯度会被保留。 |\n",
    "| `.backward()` | 从当前张量开始反向传播，计算梯度。 |\n",
    "| `.grad_fn` | 指向创建该张量的函数（非叶子节点才有）。 |\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "98048d2d-6e70-400a-b55b-01e4965c12d9",
   "metadata": {},
   "source": [
    "6. Tensor 与 NumPy 互操作"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c192dc6-bcfc-481c-a6d6-ef45c2e58453",
   "metadata": {},
   "source": [
    "| 操作 | 共享内存? | 描述 |\n",
    "| :--- | :--- | :--- |\n",
    "| `torch.from_numpy(ndarray)` | 是 | 从NumPy数组创建Tensor，共享内存（需数据类型匹配）。 |\n",
    "| `torch.tensor(ndarray)` | 否 | 从NumPy数组创建Tensor，总是进行数据拷贝。 |\n",
    "| `tensor.numpy()` | 是 | 将Tensor转为NumPy数组（仅限CPU Tensor），共享内存。 |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24a96387-28b9-4d5a-af12-3192f35e7912",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:pytorch]",
   "language": "python",
   "name": "conda-env-pytorch-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
