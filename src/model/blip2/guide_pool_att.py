import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#不分多头,单头就够了,因此没有加入head
class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        # 保持原投影层不变
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_embeds, guide_embeds):
        B, C, D = query_embeds.shape  # C=32
        
        # 关键修改：移除unsqueeze操作，直接使用32通道指导向量
        q = self.q_proj(guide_embeds)  # (B,32,D) -> (B,32,D)
        k = self.k_proj(query_embeds)  # (B,32,D)
        v = self.v_proj(query_embeds)  # (B,32,D)
        
        # 计算通道间注意力
        attn_logits = torch.matmul(q, k.transpose(1, 2))  # (B,32,32)
        attn_logits = attn_logits / math.sqrt(D)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # 加权融合后保持32通道
        output = torch.matmul(attn_weights, v)  # (B,32,D)
        output = self.out_proj(output)

        return output



class Transformer(nn.Module):
    def __init__(self, embed_dim, num_layers=4, dropout=0.1):  # 添加层数参数
        super().__init__()
        self.layers = nn.ModuleList([
            self._build_layer(embed_dim, dropout) 
            for _ in range(num_layers)
        ])
    
    def _build_layer(self, embed_dim, dropout):
        # 将原单层代码包装为独立层
        return nn.ModuleDict({
            'cross_attn': MultiHeadedAttention(embed_dim),
            'linear_proj': nn.Linear(embed_dim, embed_dim),
            'norm1': nn.LayerNorm(embed_dim),
            'norm2': nn.LayerNorm(embed_dim),
            'norm3': nn.LayerNorm(embed_dim),
            'dropout': nn.Dropout(dropout)
        })
        
    def forward(self, query_embeds, guide_embeds):
        x = query_embeds
        for layer in self.layers:
            # 每层独立归一化
            x_norm = layer['norm1'](x)
            g_norm = layer['norm1'](guide_embeds)
            
            attn_out = layer['cross_attn'](x_norm, g_norm)
            attn_out = layer['norm2'](attn_out)
            
            linear_out = layer['linear_proj'](attn_out)
            out = attn_out + layer['dropout'](linear_out)
            x = layer['norm3'](out)  # 更新x作为下一层输入
            
        return x





if __name__ == '__main__':
    # 设置测试参数
    embed_dim = 256
    num_heads = 8
    batch_size = 64
    num_texts = 64
    num_frames = 32
    
    # 创建随机测试数据
    text_embeds = torch.randn(num_texts,num_frames, embed_dim) #(B ,32, DIM)
    video_embeds = torch.randn(batch_size, num_frames, embed_dim)#(B,32,DIM)
    
    # 测试 MultiHeadedAttention
    print("测试 MultiHeadedAttention 模块...")
    mha = MultiHeadedAttention(embed_dim=embed_dim)
    mha_output = mha(video_embeds, text_embeds)
    print(f"MultiHeadedAttention 输出形状: {mha_output.shape}")
    
    # 测试 Transformer
    print("\n测试 Transformer 模块...")
    transformer = Transformer(embed_dim=embed_dim , num_layers=8)
    transformer_output = transformer(video_embeds, text_embeds)#(B,B,DIM)
    print(f"Transformer 输出形状: {transformer_output.shape}")
    
    # 验证输出维度是否正确
    expected_shape = (batch_size, num_frames, embed_dim)
    assert mha_output.shape == expected_shape, f"MultiHeadedAttention 输出形状错误: 期望 {expected_shape}, 得到 {mha_output.shape}"
    assert transformer_output.shape == expected_shape, f"Transformer 输出形状错误: 期望 {expected_shape}, 得到 {transformer_output.shape}"
    
    print("\n所有测试通过！输出维度符合预期。")