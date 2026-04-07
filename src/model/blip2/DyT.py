import torch
import torch.nn as nn
from timm.models.layers import LayerNorm2d


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 创建一个简单的测试网络
    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(64)
    
    # 初始化测试网络
    test_net = TestNet()
    print("原始网络结构:")
    print(test_net)
    
    # 转换为DynamicTanh
    converted_net = convert_ln_to_dyt(test_net)
    print("\n转换后的网络结构:")
    print(converted_net)
    
    # 测试DynamicTanh的直接使用
    dyt = DynamicTanh(normalized_shape=32, channels_last=True)
    print("\nDynamicTanh参数:")
    print(f"Alpha: {dyt.alpha.item():.4f}")
    print(f"Weight shape: {dyt.weight.shape}")
    print(f"Bias shape: {dyt.bias.shape}")
    
    # 创建测试输入
    batch_size = 4
    seq_len = 8
    
    # 测试channels_last=True的情况
    x_channels_last = torch.randn(batch_size, seq_len, 32)  # [B, S, C]
    out_channels_last = dyt(x_channels_last)
    print("\nChannels Last 测试:")
    print(f"输入形状: {x_channels_last.shape}")
    print(f"输出形状: {out_channels_last.shape}")
    
    # 测试channels_last=False的情况
    dyt_channels_first = DynamicTanh(normalized_shape=32, channels_last=False)
    x_channels_first = torch.randn(batch_size, 32, 16, 16)  # [B, C, H, W]
    out_channels_first = dyt_channels_first(x_channels_first)
    print("\nChannels First 测试:")
    print(f"输入形状: {x_channels_first.shape}")
    print(f"输出形状: {out_channels_first.shape}")
    
    # 测试不同的alpha值对激活函数的影响
    # test_x = torch.linspace(-2, 2, 100)
    # dyt.alpha = nn.Parameter(torch.ones(1) * 1.0)  # 设置alpha为1.0
    # out_alpha_1 = dyt(test_x)
    # dyt.alpha = nn.Parameter(torch.ones(1) * 2.0)  # 设置alpha为2.0
    # out_alpha_2 = dyt(test_x)
    
    # print("\n不同alpha值的影响:")
    # print(f"Alpha=1.0时的输出范围: [{out_alpha_1.min().item():.4f}, {out_alpha_1.max().item():.4f}]")
    # print(f"Alpha=2.0时的输出范围: [{out_alpha_2.min().item():.4f}, {out_alpha_2.max().item():.4f}]")