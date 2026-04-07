import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, seq1, seq2):
        num_vids, seq1_len, _ = seq1.shape
        num_vids, seq2_len, _ = seq2.shape
    
        # Query投影和reshape (来自seq1)
        q = self.q_proj(seq1)  # [batch_size, seq1_len, embed_dim]
        q = q.reshape(num_vids, seq1_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq1_len, head_dim]
    
        # Key投影和reshape (来自seq2)
        k = self.k_proj(seq2)  # [batch_size, seq2_len, embed_dim]
        k = k.reshape(num_vids, seq2_len, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq2_len, head_dim]
    
        # Value投影和reshape (来自seq2)
        v = self.v_proj(seq2)  # [batch_size, seq2_len, embed_dim]
        v = v.reshape(num_vids, seq2_len, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq2_len, head_dim]
    
        # 计算注意力分数
        attention_logits = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq1_len, seq2_len]
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=-1)
    
        # 应用注意力
        attention = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq1_len, head_dim]
        attention = attention.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq1_len, num_heads, head_dim]
        attention = attention.reshape(num_vids, seq1_len, self.embed_dim)  # [batch_size, seq1_len, embed_dim]
    
        # 输出投影
        o = self.out_proj(attention)  # [batch_size, seq1_len, embed_dim]
        return o

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=3, dropout=0.1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadedAttention(embed_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'linear': nn.Linear(embed_dim, embed_dim)
            }) for _ in range(num_layers)
        ])
        
    def forward(self, topk_video_pooled, video_embeds):
        x = video_embeds
        for layer in self.layers:
            # 多头注意力
            attn_out = layer['attn'](topk_video_pooled, x)
            attn_out = layer['norm1'](attn_out)
            
            # 前馈网络
            linear_out = layer['linear'](attn_out)
            x = attn_out + layer['norm2'](linear_out)
            
        return x


if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 设置模型参数
    embed_dim = 768  # 与BERT输出维度一致
    num_heads = 8
    batch_size = 4
    num_frames = 32  # 假设每个视频有32帧
    
    # 初始化Transformer模型
    transformer = Transformer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=3,  # 明确指定层数
        dropout=0.1
    )
    
    # 创建模拟输入
    # 全局特征向量 (也是一个序列)
    topk_video_pooled = torch.randn(batch_size,num_frames, embed_dim)
    
    # 帧级特征序列
    video_embeds = torch.randn(batch_size, num_frames, embed_dim)
    
    print("输入特征形状:")
    print(f"全局特征形状: {topk_video_pooled.shape}")
    print(f"帧序列特征形状: {video_embeds.shape}")
    
    # 进行特征融合
    try:
        fused_feat = transformer(topk_video_pooled, video_embeds)
        print("\n特征融合成功!")
        print(f"融合后特征形状: {fused_feat.shape}")
        
        # 检查输出是否符合预期
        expected_shape = (batch_size, num_frames, embed_dim)
        assert fused_feat.shape == expected_shape, f"输出形状与预期不符，期望 {expected_shape} 但得到 {fused_feat.shape}"
        print("输出形状验证通过!")
        
        # 检查输出是否包含有效值
        print(f"\n输出特征的统计信息:")
        print(f"平均值: {fused_feat.mean().item():.4f}")
        print(f"标准差: {fused_feat.std().item():.4f}")
        print(f"最小值: {fused_feat.min().item():.4f}")
        print(f"最大值: {fused_feat.max().item():.4f}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")