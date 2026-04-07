import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.blip2.xpool_cross_att import Transformer

class SubspaceTransformer(Transformer):
    def __init__(self, embed_dim, num_heads, subspace_dim=8, dropout=0.1):
        """
        新增参数：
        subspace_dim: 子空间维度k (默认8)
        """
        super().__init__(embed_dim, num_heads, dropout)
        self.subspace_dim = subspace_dim
        
        # 动态生成子空间基向量的网络
        self.subspace_generator = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, subspace_dim * embed_dim)
        )
        
        # 正交化处理层
        self.ortho_norm = nn.LayerNorm(embed_dim)

    def get_query_subspace(self, text_embeds):
        """
        生成正交子空间基向量
        输入: text_embeds (B, dim)
        输出: U (B, k, dim)
        """
        B, dim = text_embeds.shape
        
        # 生成原始基向量 (B, k*dim)
        raw_bases = self.subspace_generator(text_embeds)
        
        # 重塑为 (B, k, dim)
        U = raw_bases.view(B, self.subspace_dim, dim)
        
        # 正交化处理 (Gram-Schmidt)
        U_ortho = []
        for i in range(B):
            # 对每个查询独立正交化
            basis = self.gram_schmidt(U[i])  # (k, dim)
            U_ortho.append(basis)
        
        return torch.stack(U_ortho)  # (B, k, dim)

    def gram_schmidt(self, vectors):
        """ Gram-Schmidt正交化 """
        basis = []
        for v in vectors:
            w = v - sum(torch.dot(v, b) * b for b in basis)
            if w.norm() > 1e-8:  # 避免零向量
                w = F.normalize(w, p=2, dim=0)
                basis.append(w)
        return torch.stack(basis) if basis else vectors

    def forward(self, text_embeds, video_embeds):
        """
        增强版前向传播：
        1. 原始Transformer处理
        2. 子空间投影降维
        """
        # 原始Transformer输出 (B, B, dim)
        attn_out = super().forward(text_embeds, video_embeds)
        attn_out = F.normalize(attn_out, dim=-1)
        
        # 生成查询相关的子空间基 (B, k, dim)
        U = self.get_query_subspace(text_embeds)
        
        # 正交归一化
        U = self.ortho_norm(U)
        
        # 投影计算 (高效批量操作)
        # 投影系数: (B, B, k) = einsum('bki,bji->bjk', U, attn_out)
        proj_coeff = torch.einsum('bki,bji->bjk', U, attn_out)
        
        # 重构视频表示: (B, B, dim) = einsum('bki,bjk->bji', U, proj_coeff)
        rec_vids = torch.einsum('bki,bjk->bji', U, proj_coeff)

        rec_vids = F.normalize(rec_vids, dim=-1)
        
        return rec_vids  # (B, B, dim)

if __name__ == "__main__":
    # 参数设置
    embed_dim = 256
    num_heads = 8
    B = 64
    subspace_dim = 8
    
    # 初始化模型
    model = SubspaceTransformer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        subspace_dim=subspace_dim
    )
    
    # 生成随机测试数据
    torch.manual_seed(42)  # 固定随机种子
    text_embeds = torch.randn(B, embed_dim)  # (64, 256)
    video_embeds = torch.randn(B, 1, embed_dim)  # (64, 1, 256)
    
    print("输入形状验证:")
    print(f"文本嵌入: {text_embeds.shape}")
    print(f"视频嵌入: {video_embeds.shape}")
    
    # 前向传播
    output = model(text_embeds, video_embeds)
    
    print("\n输出形状验证:")
    print(f"输出张量: {output.shape}")  # 应该得到 (64, 64, 256)
    
    # 简单数值检查
    print("\n数值检查:")
    print(f"输出均值: {output.mean().item():.4f}")
    print(f"输出标准差: {output.std().item():.4f}")
    print(f"NaN值检测: {torch.isnan(output).any()}")
    
    # 子空间维度验证
    U = model.get_query_subspace(text_embeds)
    print(f"\n子空间基形状: {U.shape}")  # 应该得到 (64, 8, 256)
    
    # 正交性检查 (随机抽样检查一个batch)
    sample_idx = 0
    basis = U[sample_idx]  # (8, 256)
    dot_matrix = torch.mm(basis, basis.T)  # (8,8)
    eye_matrix = torch.eye(subspace_dim, device=basis.device)
    ortho_error = torch.norm(dot_matrix - eye_matrix)
    print(f"\n正交性误差 (样本{sample_idx}): {ortho_error:.4f}")
