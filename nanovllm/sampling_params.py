from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    # False（默认）：遇到EOS token时停止生成
    # True：忽略EOS token，继续生成直到达到max_tokens限制
    ignore_eos: bool = False
