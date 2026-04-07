
import torch
import torch.nn.functional as F


def guide_pool(features, guide_vector):
    """
    features: (B, 32, D) 特征张量
    guide_vector: (B, D) 指导向量
    返回: (B, D) 池化结果
    """
    # 1️⃣ 计算余弦相似度矩阵
    # 归一化保证纯方向比较
    features_norm = F.normalize(features, p=2, dim=-1)  # (B,32,D)
    guide_norm = F.normalize(guide_vector, p=2, dim=-1) # (B,D)
    
    # 2️⃣ 扩展指导向量维度
    guide_expanded = guide_norm.unsqueeze(1)  # (B,1,D)
    
    # 3️⃣ 计算相似度得分
    similarity = torch.sum(features_norm * guide_expanded, dim=-1)  # (B,32)
    
    # 4️⃣ 选取最相似通道索引
    _, indices = torch.max(similarity, dim=1)  # (B,)
    
    # 5️⃣ 根据索引提取通道
    return features[torch.arange(features.size(0)), indices]  # (B,D)