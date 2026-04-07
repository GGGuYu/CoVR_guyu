# cross_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# 多头交叉注意力（无 mask）
# --------------------------------------------------
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        """
        x1: (B, N1, dim)  -> Query
        x2: (B, N2, dim)  -> Key / Value
        return (B, N1, dim)
        """
        B, N1, _ = x1.shape
        B, N2, _ = x2.shape

        # 线性映射
        q = self.q_proj(x1)  # (B, N1, dim)
        k = self.k_proj(x2)  # (B, N2, dim)
        v = self.v_proj(x2)  # (B, N2, dim)

        # 拆多头
        q = q.view(B, N1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N1, d)
        k = k.view(B, N2, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N2, d)
        v = v.view(B, N2, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N2, d)

        # 注意力
        scores = (q @ k.transpose(-2, -1)) * self.scale        # (B, h, N1, N2)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                         # (B, h, N1, d)

        # 合并头
        out = out.transpose(1, 2).contiguous().view(B, N1, -1)  # (B, N1, dim)
        out = self.out_proj(out)
        return out

# --------------------------------------------------
# 带 CLS 的交叉 Transformer
# --------------------------------------------------
class CrossTransformer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        """
        x1: (B, N1, dim)
        x2: (B, N2, dim)
        return (out, cls)
            out: (B, 1+N1, dim)  带 CLS 的完整序列
            cls: (B, dim)        单独的 CLS token
        """
        B = x1.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B,1,dim)

        # 在 x1 前面拼 CLS
        x1_with_cls = torch.cat([cls_tokens, x1], dim=1)   # (B, 1+N1, dim)
        x1_with_cls = self.norm1(x1_with_cls)

        # 交叉注意力: Query = x1_with_cls,  K/V = x2
        attn_out = self.cross_attn(x1_with_cls, x2)
        x = x1_with_cls + attn_out
        x = self.norm2(x)

        # FFN
        x = x + self.ffn(x)

        # 拆分返回
        cls = x[:, 0, :]          # (B, dim)
        out = x                   # (B, 1+N1, dim)
        return cls           # 按需取用即可

# --------------------------------------------------
# 快速自测
# --------------------------------------------------
if __name__ == "__main__":
    B, N1, N2, dim, h = 4, 7, 10, 128, 8
    x1 = torch.randn(B, N1, dim)
    x2 = torch.randn(B, N2, dim)

    model = CrossTransformer(dim, h)
    out, cls = model(x1, x2)

    print("out shape:", out.shape)   # (B, 1+N1, dim)
    print("cls shape:", cls.shape)   # (B, dim)