import os
from dataclasses import dataclass
from transformers import AutoConfig

# 方便打印， 格式化
# 用来保存运行时配置的类 Config，使用 dataclass 自动生成构造函数和便于打印的 repr

# 使用 dataclass 装饰器，自动生成 init、repr 等
@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # 所有请求的所有token总行
    max_num_seqs: int = 512             # 同时处理的最大序列（请求）数量
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9 # 用于计算 KV 缓存块数量
    tensor_parallel_size: int = 1
    enforce_eager: bool = False         # 禁用 CUDA 图优化
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1        # -1  代表自动结算

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 读取的是模型所在文件夹里config.json里的内容
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
