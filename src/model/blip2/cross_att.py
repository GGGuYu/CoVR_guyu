import torch
import torch.nn as nn
import torch.nn.functional as F # 导入 F

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # 存储 dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        # 将输入视为长度为 1 的序列 [bs, 1, dim]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        q = self.q_proj(x1) # [bs, 1, dim]
        k = self.k_proj(x2) # [bs, 1, dim]
        v = self.v_proj(x2) # [bs, 1, dim]

        # 计算注意力分数 [bs, 1, 1]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim**0.5) # 使用 self.dim
        attn = F.softmax(attn_scores, dim=-1)

        # 应用注意力权重到 V [bs, 1, dim]
        output = torch.matmul(attn, v)

        # 返回 [bs, dim]
        return output.squeeze(1)