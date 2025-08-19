from dataclasses import dataclass
import torch
# 这个文件就是全局变量， 并不是传统意义上的上下文
# 实现了一个全局状态管理系统，用于在 nano-vllm 推理过程中协调各个组件之间的参数传递
# 使得推理过程中的各个组件能够共享关键的执行参数
# 这种设计避免了在函数调用链中传递大量参数，简化了代码结构，特别适合深度学习推理场景中的复杂参数协调需求

# Context 类封装了推理执行期间需要的所有关键参数：
@dataclass
class Context:
    is_prefill: bool = False                        # 标识当前是预填充阶段还是解码阶段
    cu_seqlens_q: torch.Tensor | None = None        # FlashAttention 需要的累积序列长度
    cu_seqlens_k: torch.Tensor | None = None        # FlashAttention 需要的累积序列长度
    max_seqlen_q: int = 0                           # 批次中的最大序列长度
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None        # 序列位置到 KV 缓存内存槽的映射
    context_lens: torch.Tensor | None = None        # 每个序列的上下文长度
    block_tables: torch.Tensor | None = None        # 每个序列的 KV 缓存块表映射

_CONTEXT = Context()

# 获取当前全局上下文实例
def get_context():    
    return _CONTEXT # 读取全局变量不需要 global，只有“赋值/重绑定”时才需要

# 更新全局上下文参数
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

# 重置上下文为默认值
def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
