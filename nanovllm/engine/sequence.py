# 管理单个推理请求的核心数据结构，包含序列状态、token 数据和内存管理功能
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)            # 分配唯一序列 ID，类属性，所有实例共享，只能保证单一进程内的唯一性
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)                # 存储的 token 序列，后续会在这里追加新生成的 token
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)           # 变长的，随着decode增加
        self.num_prompt_tokens = len(token_ids)         # 定长的，提示词的长度
        self.num_cached_tokens = 0                      # 已计算 KV 缓存的 token 数量，只在内存管理器分配 KV 缓存块时更新，用于优化预填充阶段，避免重复计算已缓存的 token
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    # 把一个无参数的方法“包装”成只读属性，访问时不用加括号
    # 适合把“计算得到的值”当作字段暴露，API 更自然且不可被随意改写
    # 可读性高，且默认只读，防止外部随意赋值破坏一致性
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):                        # 返回生成的 token 数量
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size # 计算已缓存的完整块数量

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size # 计算序列需要的总块数， 向上取整的关键技巧

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size # 计算最后一个块中的 token 数量

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]  # 提取指定块的 token 数据

    def append_token(self, token_id: int): # 向序列添加新生成的 token
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 自定义序列化机制允许 Sequence 对象在分布式推理中高效传输，只保存必要的状态信息而不是完整对象，减少了内存使用和传输开销
    # 这种自定义序列化机制允许 Sequence 对象在分布式推理中高效传输，
    # 只保存必要的状态信息而不是完整对象，减少了内存使用和传输开销。这在多进程并行推理场景中特别重要
    # 定义对象序列化时要保存的状态数据
    def __getstate__(self): 
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)            

    # 定义对象反序列化时如何恢复状态
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]                                                        
