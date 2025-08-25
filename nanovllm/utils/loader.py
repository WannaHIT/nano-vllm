import os
from glob import glob                   # 用于匹配目录下所有 .safetensors 文件
import torch
from torch import nn
from safetensors import safe_open       # 只按需映射读取 tensor（不一次性全加载），第三个参数是设备（"cpu"）


# 就地复制到已有的参数内存，不创建新对象，保持参数引用关系
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


"""
双层 for + else:实现“尝试特殊映射 → 否则默认加载”。
packed_modules_mapping 用于“文件名片段 → (真实参数前缀, shard_id)”的重写与分片拼装。
weight_loader 是挂在单个 nn.Parameter 上的回调，支持：
分片累积（例如多次调用把各 shard 写入不同切片）
量化反量化或 reshape
默认 loader 只做一次 copy_。
safe_open(..., "pt", "cpu")：第二个参数格式标记，第三个设备，不移动到 GPU(后续统一 to(device))
"""
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})   # 读取模型上可选属性
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():                                    # f.keys() 列出文件里存储的所有张量名字（字符串）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # v 是要替换成的真实参数前缀，shard_id 给自定义加载器（表示第几片或哪一维的分片）
                        v, shard_id = packed_modules_mapping[k]
                        # 构造真实参数名
                        param_name = weight_name.replace(k, v)
                        # 拿到 nn.Parameter
                        param = model.get_parameter(param_name)
                        # 要求该参数对象已挂载自定义属性 weight_loader（不是标准 PyTorch 接口，说明外部初始化时加过）
                        weight_loader = getattr(param, "weight_loader")
                        # 自定义逻辑（做拼接、切片写入、解打包）
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 直接按原名字获取参数
                    param = model.get_parameter(weight_name)
                    # 尝试取 param.weight_loader；若没有，则用默认的 default_weight_loader
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))     # 加载对应张量并调用 loader
