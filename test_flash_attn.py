import torch
from flash_attn import flash_attn_func

# 检查是否有可用的CUDA设备
if not torch.cuda.is_available():
    print("错误: 未检测到可用的CUDA GPU。FlashAttention无法运行。")
else:
    print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建一些需要在GPU上的虚拟数据
    # 尺寸: (批大小, 头数量, 序列长度, 头维度)
    q = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)

    print("测试数据已创建,准备调用FlashAttention核心函数...")

    # 调用flash_attn的核心函数
    try:
        output = flash_attn_func(q, k, v)
        print("\n 恭喜! FlashAttention 核心函数运行成功！")
        print(f"输出张量的形状 (Output shape): {output.shape}")
        print(f"输出张量的数据类型 (Output dtype): {output.dtype}")
    except Exception as e:
        print("\n 出错了! 运行FlashAttention核心函数时遇到问题: ")
        print(e)
