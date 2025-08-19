from nanovllm.engine.block_manager import BlockManager

# 序列 A：块1=[1,2,3]，块2=[4,5,6]，块3=[7,8,9]
# 序列 B：块1=[1,2,3]，块2=[4,5,6]，块3=[7,8,0] 
# 只有“到当前块为止的整段前缀”完全相同，哈希才会相同
b1 = [1,2,3]
b2 = [4,5,6]
b3a = [7,8,9]
b3b = [7,8,0]

h1 = BlockManager.compute_hash(b1)          # prefix=-1
h2 = BlockManager.compute_hash(b2, h1)       # 链到上一块
h3a = BlockManager.compute_hash(b3a, h2)     # 继续链
h3b = BlockManager.compute_hash(b3b, h2)     # 同样链到相同前缀

assert h3a!=h3b                              # 不同的第三块得到不同哈希