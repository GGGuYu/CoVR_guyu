import torch
import torch.nn.functional as F

def guide_topk_pool(features, guide_vector, k=3):
    """
    选择与指导向量最相似的 top-k 通道
    
    Args:
        features: (B, 32, D) 特征张量
        guide_vector: (B, D) 指导向量
        k: 选择相似度最高的前k个通道
        
    Returns:
        topk_features: (B, k, D) 最相似的k个通道
        topk_indices: (B, k) 被选中的通道索引
    """
    # 1️⃣ 归一化保证纯方向比较
    features_norm = F.normalize(features, p=2, dim=-1)  # (B, 32, D)
    guide_norm = F.normalize(guide_vector, p=2, dim=-1)  # (B, D)
    
    # 2️⃣ 扩展指导向量维度
    guide_expanded = guide_norm.unsqueeze(1)  # (B, 1, D)
    
    # 3️⃣ 计算余弦相似度
    similarity = torch.sum(features_norm * guide_expanded, dim=-1)  # (B, 32)
    
    # 4️⃣ 获取top-k相似度通道
    topk_scores, topk_indices = torch.topk(similarity, k=k, dim=1)  # (B, k)
    
    # 5️⃣ 收集对应的特征通道
    batch_indices = torch.arange(features.size(0))[:, None]  # (B, 1)
    topk_features = features[batch_indices, topk_indices]    # (B, k, D)
    
    return topk_features, topk_indices
