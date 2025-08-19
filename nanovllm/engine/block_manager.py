from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id            # uid
        self.ref_count = 0                  # prefix相同阶段 共同索引
        self.hash = -1                      # 内容哈希缓存和去重， 通过哈系值快速比对两个block是否相同
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []                 # 清空token_ids


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]    # i是block_id
        # key 是块的链式哈希值，value 是块ID。
        # 用于“内容去重”：先哈希快速命中，再比对 token_ids 确认。
        self.hash_to_block_id: dict[int, int] = dict()                      # 哈希值到块ID的映射，用于内容去重
        self.free_block_ids: deque[int] = deque(range(num_blocks))          # 可用块ID的队列,新分配时从这里取一个ID；释放时把ID放回队尾
        self.used_block_ids: set[int] = set()                               # 已使用块ID的集合,用于判断某个块是否已被占用（以及共享复用时递增 ref_count）

    # 用于“前缀缓存/去重”。当块满了才计算哈希，未满块在调用处用 -1 表示不参与去重
    # prefix 上一块（或上一阶段）的哈希，默认 -1 表示没有前缀
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):          # 一个块放满了才计算哈系
        h = xxhash.xxh64()                              # 哈希器只能吃字节流                                       # 创建 xxh64 哈希器
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))      # 把 prefix 转成 8 字节小端并先喂给哈希器（把“前缀链”纳入当前块的哈希）
        h.update(np.array(token_ids).tobytes())                             # 再把当前块的 token_ids 转成 连续的字节流 numpy 再转原始字节喂给哈希器
        return h.intdigest()                            # 返回 64 位整型哈希值

    """
    分配指定ID的物理块
    """
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()                               # ref_count重置为1、清空内容、hash = -1
        self.free_block_ids.remove(block_id)        # 弹出
        self.used_block_ids.add(block_id)           # 加入
        return self.blocks[block_id]

    """
    回收指定ID的物理块
    """
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0 # 要求 ref_count == 0（最后一个引用已释放）
        self.used_block_ids.remove(block_id)        # 从 used_block_ids 移除
        self.free_block_ids.append(block_id)        # 放回 free_block_ids

    """
    预检查空闲块数量是否足够容纳序列seq需要的块数
    """
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    """
    为一个新序列建立块表并进行去重复用
    """
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 未满块“不参与去重”
            # 只有“链式哈希一致”且“token_ids 完全相等”才算可复用
            # 只给“满块”算哈希；不满块直接设 h = -1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1    # 计算哈希：为每个完整块计算哈希值
            # 查是否命中
            # 去重加速（哈希命中 + 内容校验 + 复用）
            # 先用哈希表 O(1) 查重；命中后再比对内容避免碰撞
            block_id = self.hash_to_block_id.get(h, -1)         # 先用链式哈希命中表, 查找重复：通过 hash_to_block_id 映射查找是否已存在相同内容的块
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:  # 视为不相同（防止哈希碰撞）
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            # 命中（非 cache_miss）时复用已有块：不再分配新块，增加 ref_count，并计入已缓存 token 数
            else:                                                               # 共享复用：如果找到相同哈希的块，直接复用而不分配新块
                seq.num_cached_tokens += self.block_size                        # 标记这些 token 不需要重新计算,每次增加一个完整块的大小
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            # 只有 h != -1（即满块） 才会更新块的哈希与索引表
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    """
    释放这个序列占用的所有 KV 缓存块，并把块 ID 放回可用队列；
    同时清空序列的块表和已缓存 token 计数
    """
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):          # 逆序遍历块表，从最后一块开始释放
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                # hash_to_block_id 映射不会删，保留用于后续内容复用（命中后仍会二次比对 token_ids 防碰撞）
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0       # 清零该序列“已缓存 token 数”，后续不再认为它有可复用的 KV
        seq.block_table.clear()         # 清空块表，这个序列不再持有任何块

    """
    追加一个token前的资源检查
    """
    def can_append(self, seq: Sequence) -> bool:
        # 只有在“即将开启新块”（长度模 block_size 等于 1）时才需要检查至少有1个空闲块，否则不需要新块
        # len(seq)=8 → 8%4=0 → 右边=False(0) → 需要0块 → 条件恒为真
        # len(seq)=9 → 9%4=1 → 右边=True(1) → 需要1块 → 只有有≥1个空闲块才返回 True
        # 这个 1 代表 刚刚开始一个新的块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    """
    追加阶段的块级账本更新
    """
    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # "即将"开始一个新块，需要先为它分配一个空块
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1           # 检测上一个块是满的，且有hash
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # 只有当最后一块“恰好写满”才计算哈希；计算该块的链式哈希并登记到 hash_to_block_id
        # 刚写满时还没计算哈希，也应当是 -1
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            # 取“最后一块”的 token 切片用于计算哈希
            token_ids = seq.block(seq.num_blocks-1)     # block(i)通常按 i*self.block_size 计算起止位置，要求非负索引
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)             # 更新
            self.hash_to_block_id[h] = last_block.block_id # 添加映射
        # 未满块不参与去重，hash 必须是 -1
        else:
            assert last_block.hash == -1
