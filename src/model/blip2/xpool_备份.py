import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim , num_heads ,dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # ===== 新增：eps和全局平均熵 =====
        self.eps = 1e-8  # 防止log(0)
        self.attn_entropy = 0
        # ============================

    
    def forward(self, text_embeds, video_embeds,is_training = False):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)
        # attention_weights = F.softmax(attention_logits, dim=-1)
        # print("attention_weights:",attention_weights)

        # ===== 新增：计算注意力熵 =====
        if is_training:  # 仅在训练时计算
            # 计算每个注意力分布的熵 (num_vids, num_heads, num_texts)
            entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + self.eps), 
                dim=2
            )
            # 全局平均熵 (标量)
            self.attn_entropy = entropy.mean() 
        # ============================


        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class Transformer(nn.Module):
    def __init__(self, embed_dim , num_heads , dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        dropout = dropout

        self.cross_attn = MultiHeadedAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds,is_training = False):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds,is_training)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        # ===== 修改：获取注意力熵 =====
        if is_training:
            attn_entropy = self.cross_attn.attn_entropy  # 从MHA获取熵值
        return (out, attn_entropy) if is_training else out
        # ============================

        
        return out

"""
输入
(B , dim) 相当于你的文本嵌入
(B , B , dim) 交叉结果
pooling_type = 'avg'
输出是 (B , B)
"""
def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)#转化正确的形状 num_texts x embed_dim x num_vids
        # print(f'vid_embeds_pooledu反转后的形状:{vid_embeds_pooled.shape}')
        text_embeds = text_embeds.unsqueeze(1)# num_texts x 1 x embed_dim
        # print(f'text_embeds的形状:{text_embeds.shape}')
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1) #得到B,B相似度矩阵，是每个查询和特化视频库的相似度矩阵

        # sims = torch.mm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims


def cirr_sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)#转化正确的形状 num_texts x embed_dim x num_vids
        
        text_embeds = text_embeds.unsqueeze(1)# num_texts x 1 x embed_dim
        
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1) #得到B,B相似度矩阵，是每个查询和特化视频库的相似度矩阵

    return sims

if __name__ == '__main__':
    # 设置测试参数
    embed_dim = 256
    num_heads = 8
    batch_size = 80
    num_texts = 32
    num_frames = 1
    
    # 创建随机测试数据
    text_embeds = torch.randn(num_texts, embed_dim) #(B , DIM)
    video_embeds = torch.randn(batch_size, num_frames, embed_dim)#(B,1,DIM)
    
    # 测试 MultiHeadedAttention
    print("测试 MultiHeadedAttention 模块...")
    mha = MultiHeadedAttention(embed_dim=embed_dim, num_heads=num_heads)
    mha_output = mha(text_embeds, video_embeds)
    print(f"MultiHeadedAttention 输出形状: {mha_output.shape}")
    
    # 测试 Transformer
    print("\n测试 Transformer 模块...")
    transformer = Transformer(embed_dim=embed_dim, num_heads=num_heads)
    transformer_output = transformer(text_embeds, video_embeds)#(B,B,DIM)
    print(f"Transformer 输出形状: {transformer_output.shape}")
    
    # 验证输出维度是否正确
    expected_shape = (batch_size, num_texts, embed_dim)
    assert mha_output.shape == expected_shape, f"MultiHeadedAttention 输出形状错误: 期望 {expected_shape}, 得到 {mha_output.shape}"
    assert transformer_output.shape == expected_shape, f"Transformer 输出形状错误: 期望 {expected_shape}, 得到 {transformer_output.shape}"
    
    print("\n所有测试通过！输出维度符合预期。")
    
    # 新增 sim_matrix_training 测试
    print("\n测试 sim_matrix_training 模块...")
    # 创建测试数据
    text_embeds = torch.randn(num_texts, embed_dim)#(B,DIM)
    vid_embeds_pooled = torch.randn(batch_size, num_texts, embed_dim)#(B,B,DIM)
    
    
    # 测试 xpool
    xpool_sims = sim_matrix_training(text_embeds, vid_embeds_pooled, 'xpool')
    print(f"xpool 输出形状: {xpool_sims.shape}") 
    assert xpool_sims.shape == (num_texts, batch_size), f"xpool 形状错误: 期望 {(batch_size, batch_size)}, 得到 {xpool_sims.shape}"
    
    print("\nsim_matrix_training 测试通过！输出维度符合预期。")