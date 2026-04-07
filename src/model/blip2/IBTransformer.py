import torch
import torch.nn as nn
from xpool_cross_att import Transformer


class IB_Specialized_Library(nn.Module):
    """
    使用信息瓶颈理论来生成特化视频库的模块。
    
    这个模块会包裹一个基础的 Transformer 模型，并在其之上增加 VIB 的逻辑。
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # 1. 基础的交叉注意力模块，用于生成初步的交互特征
        self.base_transformer = Transformer(embed_dim, num_heads, dropout)
        
        # 2. 两个线性层，分别用于预测高斯分布的 mu (均值) 和 logvar (对数方差)
        # 输入是 (B, B, dim)，输出也是 (B, B, dim)
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从 N(mu, var) 中采样，同时保持梯度可传导。
        """
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)    # 从标准正态分布中采样噪声
        return mu + eps * std          # 采样得到特化特征

    def forward(self, text_embeds, video_embeds):
        """
        输入:
            text_embeds:  (B, dim) 的查询嵌入
            video_embeds: (B, 1, dim) 的原始视频库（答案）
        
        输出:
            specialized_vids: (B, B, dim) 的特化视频库
            kl_loss: 该模块产生的 KL 散度损失 (一个标量)
        """
        # video_embeds 形状是 (B, 1, dim)，需要扩展以匹配 text_embeds
        # 这里的 video_embeds 实际上代表了整个批次的视频库，我们把它看作 (B, dim)
        video_library = video_embeds.squeeze(1) # -> (B, dim)
        
        # 1. 首先，像原来一样，让每个查询与整个视频库进行交互
        # 注意：这里我们需要对 video_library 进行扩展，以进行 BxB 的交互
        # 我们可以将 video_library 扩展为 (B, B, dim) 让每个查询都能看到完整的库
        # 但更高效的方式是直接在 Transformer 内部处理 B 个查询和 B 个视频的交叉注意力
        # 从您原始代码的 forward pass 看，它已经能处理 num_texts vs num_vids
        # transformer_output 的形状已经是 (B, B, dim)
        # text_embeds: (B, dim) -> num_texts=B
        # video_embeds: (B, 1, dim) -> num_vids=B, num_frames=1
        base_output = self.base_transformer(text_embeds, video_embeds) # -> (B, B, dim)
        base_output = base_output.permute(1,2,0)
        # 2. 从 Transformer 输出中预测分布参数
        mu = self.fc_mu(base_output)            # -> (B, B, dim)
        logvar = self.fc_logvar(base_output)    # -> (B, B, dim)
        
        # 3. 使用重参数化技巧采样，得到特化视频库
        specialized_vids = self.reparameterize(mu, logvar) # -> (B, B, dim)
        
        # 4. 计算 KL 散度损失 (信息瓶颈正则项)
        # KL(q(z|x) || p(z)) where p(z) is N(0,1)
        # 公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 对 batch size 和库大小进行平均，使其不依赖于 B 的大小
        kl_loss = kl_loss / (mu.size(0) * mu.size(1))
        
        return specialized_vids, kl_loss