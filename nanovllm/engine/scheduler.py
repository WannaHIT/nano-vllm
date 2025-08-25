from collections import deque                   # 使用双端队列，O(1) 头尾进出队，适合等待/运行队列

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


"""
两阶段策略：先尝试打包 prefill(只要还有等待序列),否则进入 decode。
去重收益计入 len(seq) - seq.num_cached_tokens,避免把复用 token 再算到批量 token 限制里。
解码资源不足时通过 preempt 抢占尾部序列，释放其块给当前紧缺的序列。
can_append/may_append 把“是否需要新块”和“块满登记哈希”逻辑封装，调度层只负责时机
"""
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                         # 同时调度（进入本轮 batch）的最大序列数限制
        self.max_num_batched_tokens = config.max_num_batched_tokens     # 预填充阶段所有序列 token 总数上限（防止超显存）
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # 管理 KV 缓存块分配/去重/释放
        self.waiting: deque[Sequence] = deque()                         # 等待（未进入运行状态或被抢占后）的序列队列
        self.running: deque[Sequence] = deque()                         # 正在参与解码循环或刚预填充完的序列

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 目的：选出这一轮要送进模型的一批序列 + 标记是否是 prefill
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill 预填充阶段尝试
        scheduled_seqs = []             # 收集本轮选中的序列
        num_seqs = 0                    # 当前累积计数
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs: # 预填充阶段 确保不超过最大序列数
            seq = self.waiting[0]       # 查看队首但暂不弹出
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq): # 确保既不超过序列数量限制，也不超过 token 数量限制和内存限制
                break
            num_seqs += 1
            self.block_manager.allocate(seq)                 # 为提示词分块 + 去重 + 共享复用
            # 
            # “统计当前 prefill 批真实计算负载”
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 只把“需要真正算的 token”计入（如果复用了一些块，会减少），确实需要模型进行前向的 token 数
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)    
            scheduled_seqs.append(seq)  
        # 如果拿到了任何预填充序列，本轮就是 prefill，立即返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode 只有在没有任何需要 prefill 的序列时执行（只有没有可 prefill 的才 decode）
        """
        抢占别人：释放别人 → 当前 seq 获得资源 → 本轮还能 forward(跑）。
        抢占自己：释放自己 → 当前 seq 暂停 → 本轮不能 forward,只能等下一轮重新 prefill。
        """
        while self.running and num_seqs < self.max_num_seqs:    # 解码阶段：同样限制并发序列数
            seq = self.running.popleft()                        # 取出一个运行中的序列尝试本步解码
            # 若当前追加 1 个 token 可能需要新块但没有资源，需要“抢占”别人释放资源
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 抢占（踢掉）运行队列尾部的另一条序列，释放它的块（让资源出来）
                    self.preempt(self.running.pop())
                else:
                    # 如果没有其他可抢占，只能抢占自己，然后跳出外层 while（资源不足，退出调度）
                    # 这条也不能跑了，本轮结束 decode 
                    # 把自己放回 waiting
                    self.preempt(seq)
                    break
            # else 只在 while 正常结束（没 break）时执行
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)  # 在新块第一 token 时分配块
                scheduled_seqs.append(seq)          # 在块刚满时计算哈希登记
        assert scheduled_seqs                       # 确认至少安排了一条（否则逻辑有 bug）
        # 把刚调度的序列重新放回 running 的左侧，保持原先顺序（本轮仍在运行集合里，下一轮继续解码）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False                # 解码阶段返回

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)      # 释放它的所有块（引用计数递减；归还空闲块）
        self.waiting.appendleft(seq)            # 放回等待队列头部（优先级较高，之后会先做 prefill）

    """
    输入:seqs(本轮模型跑过的序列)
    token_ids(解码阶段返回的新 token 列表；预填充阶段通常为空或只用最后一个位置的采样结果)
    """
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # 把新 token 加入序列（更新 num_tokens / last_token 等）
            seq.append_token(token_id)                
            # 生成到 EOS 且不忽略，或达到用户设定最大生成长度      
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
