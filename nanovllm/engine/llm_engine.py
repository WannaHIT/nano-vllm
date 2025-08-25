import atexit                                               # 注册程序退出时的清理函数
from dataclasses import fields                              # 动态获取 Config dataclass 的字段集合，用于过滤 kwargs
from time import perf_counter                               # 高精度计时，用于吞吐统计
from tqdm.auto import tqdm
from transformers import AutoTokenizer                      # 加载 HuggingFace 分词器
import torch.multiprocessing as mp                          # 多进程（张量并行进程）
# 多进程：为张量并行 rank>0 启多个子进程，主进程 rank=0
# 调度分两阶段：prefill（批量长提示）与 decode（并行逐步生成），两者统计方式不同

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 防御式，只允许 Config 支持的参数
        config_fields = {field.name for field in fields(Config)}                    # 取出 Config dataclass 的所有字段名字集合
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}     # 只保留用户传入的、Config 支持的参数（忽略多余键）
        config = Config(model, **config_kwargs)                                     # 实例化 Config（会在其 post_init 中做校验与加载模型配置）
        self.ps = []                                                                # 保存子进程对象和进程间事件列表
        self.events = []
        ctx = mp.get_context("spawn")                                               # 指定使用 spawn 启动方式（更干净，兼容性好）
        # 主进程用 rank 0，这里为其余并行分片创建子进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()                                                     # 进程间事件，用于同步
            process = ctx.Process(target=ModelRunner, args=(config, i, event))      # 每个子进程运行一个 ModelRunner 实例
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)                     # 主进程内构建 rank=0 的模型执行器，并把所有事件列表传入
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True) 
        config.eos = self.tokenizer.eos_token_id                                    # # 把分词器的 EOS id 写回配置，供后续终止判断
        self.scheduler = Scheduler(config)                                          # 调度器：管理序列状态、分配/回收 KV 块、决定 prefill 还是 decode
        atexit.register(self.exit)                                                  # 注册退出钩子，确保进程与资源清理

    def exit(self):
        self.model_runner.call("exit")      # 通知主模型执行器做清理
        del self.model_runner               # 释放引用（促使其内部资源释放）
        for p in self.ps:                   # 等待所有子进程结束（防止孤儿进程）
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):                 # 字符串转成 token id 列表
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)     # 创建序列对象，保存初始 token
        self.scheduler.add(seq)                     # 交给调度器登记，稍后统一调度

    def step(self):
        # 调度器返回本步要跑的序列列表以及当前阶段类型（预填充或解码）
        seqs, is_prefill = self.scheduler.schedule()
        # 运行模型：预填充批或解码批，返回新生成 token（解码时）
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 根据结果更新每个序列（追加 token、状态迁移、KV 管理）
        self.scheduler.postprocess(seqs, token_ids)
        # 收集本步刚完成的序列的输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 统计吞吐：预填充阶段统计总 token 数；解码阶段用负的序列数表示本步生成了多少个新 token（后面用来区分）
        # 解码阶段每个活跃序列本步只新增 1 个 token，因此“本步生成的新 token 数” = 序列数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        # 返回本步增量结果 + 吞吐计数（供 generate 显示）
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished() # 询问调度器是否所有序列都完成

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)  # 单个 SamplingParams 复制成列表（每个 prompt 一个）
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)                        # 批量加入请求
        outputs = {}
        prefill_throughput = decode_throughput = 0.             # 吞吐数
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()                    # 计时一轮 step
            if use_tqdm:
                # 若 num_tokens > 0（预填充）
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                # 否则（解码）
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t) # 注意取负数
                # 更新吞吐显示
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)       # 进度 +1（完成一个序列）
        # 按 seq_id 排序，转成列表（序列化输出顺序确定）
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        # 解码成文本并附带 token 列表
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
