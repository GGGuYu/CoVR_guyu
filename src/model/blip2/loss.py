import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.cloud.cloud import Cloud
from src.model.blip2.xpool_cross_att import sim_matrix_training

class CrossEntropyLoss(nn.Module):
    """
    Hard Negative NCE loss for contrastive learning.
    """

    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, tar_img_feat: torch.Tensor, query_feat: torch.Tensor, temp):
        device = tar_img_feat.device

        sim_t2q = tar_img_feat @ query_feat.T / temp
        sim_q2t = query_feat @ tar_img_feat.T / temp

        bs = sim_t2q.size(0)
        loss_t2q = F.cross_entropy(sim_t2q, torch.arange(bs, device=device))
        loss_q2t = F.cross_entropy(sim_q2t, torch.arange(bs, device=device))

        return (loss_t2q + loss_q2t) / 2


class HardNegativeNCE(nn.Module):
    """
    Hard-Negative NCE loss for contrastive learning.
    https://arxiv.org/pdf/2301.02280.pdf
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs):
        """
        Args:
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(HardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        video_embds: torch.Tensor,
        text_embds: torch.Tensor,
        temp,
    ):
        """
        Args:
            video_embds: (batch_size, video_embd_dim)
            text_embds: (batch_size, text_embd_dim)
        """
        batch_size = video_embds.size(0)
        # computation of the similarity matrix
        sim_matrix = video_embds @ text_embds.T  # (batch_size, batch_size)
        # scale the similarity matrix with the temperature
        sim_matrix = sim_matrix / temp
        sim_matrix = sim_matrix.float()

        nominator = torch.diagonal(sim_matrix)

        beta_sim = self.beta * sim_matrix
        w_v2t = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim)))
        )
        w_t2v = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim)))
        )
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))

        hn_nce_loss = (denominator_v2t - nominator).mean() + (
            denominator_t2v - nominator
        ).mean()
        return hn_nce_loss

#云模型 + 硬负对比损失
class CloudHardNegativeNCE(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs):
        """
        Args:
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(CloudHardNegativeNCE, self).__init__()  # 修正为当前类名
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        video_embds: torch.Tensor,
        text_embds: torch.Tensor,
        temp,
    ):
        """
        Args:
            video_embds: (batch_size, video_embd_dim)
            text_embds: (batch_size, text_embd_dim)
        """
        batch_size = video_embds.size(0)
        #改动
        #------------------------------------------------------
        #云模型
        print(f's视频加噪')
        cloud = Cloud(video_embds, 1 ,video_embds.shape[1])
        video_embds = cloud.get_cloud()
        #------------------------------------------------------
        # computation of the similarity matrix
        sim_matrix = video_embds @ text_embds.T  # (batch_size, batch_size)
        # scale the similarity matrix with the temperature
        sim_matrix = sim_matrix / temp
        sim_matrix = sim_matrix.float()

        nominator = torch.diagonal(sim_matrix)

        beta_sim = self.beta * sim_matrix
        w_v2t = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim)))
        )
        w_t2v = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim)))
        )
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))

        hn_nce_loss = (denominator_v2t - nominator).mean() + (
            denominator_t2v - nominator
        ).mean()
        return hn_nce_loss


# xpool_hce交叉损失
class XpoolHardNegativeNCE(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs):
        """
        Args:
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(XpoolHardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        xpool_sim_matrix : torch.Tensor,
        temp,
    ):
        """
        Args:
            xpool_sim_matrix: (batch_size,batch_size,dim)
        """
        batch_size = xpool_sim_matrix.size(0)
        # computation of the similarity matrix
        sim_matrix = xpool_sim_matrix
        # scale the similarity matrix with the temperature
        sim_matrix = sim_matrix / temp
        sim_matrix = sim_matrix.float()

        nominator = torch.diagonal(sim_matrix)

        beta_sim = self.beta * sim_matrix
        w_v2t = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim)))
        )
        w_t2v = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim)))
        )
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))

        hn_nce_loss = (denominator_v2t - nominator).mean() + (
            denominator_t2v - nominator
        ).mean()
        return hn_nce_loss


#增强版硬负样本对比损失，支持跨模态硬负样本挖掘
# class CrossModalHardNegativeNCE(nn.Module):
#     def __init__(self, alpha=1.0, beta=0.0, tau_vis=0.7, tau_title=0.5, hard_weight=2.0, **kwargs):
#         """
#         增强版硬负样本对比损失，支持跨模态硬负样本挖掘
        
#         Args:
#             alpha: 正样本缩放系数 (默认: 1.0)
#             beta: 浓度系数 (默认: 0.0)
#             tau_vis: 视觉相似度阈值 (默认: 0.7)
#             tau_title: 标题相似度阈值 (默认: 0.5)
#             hard_weight: 硬负样本权重增强系数 (默认: 2.0)
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.tau_vis = tau_vis
#         self.tau_title = tau_title
#         self.hard_weight = hard_weight

#     def forward(self, xpool_sim_matrix, vid_emb, title_emb, temp=0.07):
#         """
#         Args:
#             xpool_sim_matrix: 特化后的相似度矩阵 (batch_size, batch_size)
#             vid_emb: 视频原始BLIP编码 (batch_size, embed_dim)
#             title_emb: 视频标题BLIP编码 (batch_size, embed_dim)
#             temp: 温度系数 (默认: 0.07)
#         """
#         # print(f'当前实际运行的self.tau_vis = {self.tau_vis}') #预计为修改后的0.7而不是配置文件中的0.5
#         batch_size = xpool_sim_matrix.size(0)
#         device = xpool_sim_matrix.device
        
#         # ===== 跨模态硬负样本挖掘 =====
#         # 计算视觉相似度矩阵 (B,B)
#         sim_vis = F.cosine_similarity(
#             vid_emb.unsqueeze(1), 
#             vid_emb.unsqueeze(0), 
#             dim=-1
#         )  # (B,B)
        
#         # 计算标题相似度矩阵 (B,B)
#         sim_title = F.cosine_similarity(
#             title_emb.unsqueeze(1), 
#             title_emb.unsqueeze(0), 
#             dim=-1
#         )  # (B,B)
        
#         # 生成硬负样本掩码
#         hard_neg_mask_vis = (sim_vis > self.tau_vis) & (sim_title < self.tau_title)  # 视觉相似但标题不匹配
#         # hard_neg_mask_title = (sim_title > self.tau_title) & (sim_vis < self.tau_vis)  # 标题相似但视觉不匹配
#         # hard_neg_mask = (hard_neg_mask_vis | hard_neg_mask_title)
#         hard_neg_mask = hard_neg_mask_vis
#         hard_neg_mask = hard_neg_mask.fill_diagonal_(False)  # 排除正样本
        
#         # 构建权重矩阵
#         weight_matrix = torch.ones_like(xpool_sim_matrix, device=device)
#         weight_matrix[hard_neg_mask] += self.hard_weight  # 增强硬负样本权重
        
#         # ===== 加权相似度矩阵计算 =====
#         sim_matrix = xpool_sim_matrix / temp  # 温度缩放
        
#         # 计算指数项
#         beta_sim = self.beta * sim_matrix
#         exp_beta_sim = torch.exp(beta_sim)
        
#         # 计算分母权重 (text-to-video方向)
#         w_v2t_numerator = (batch_size - 1) * exp_beta_sim
#         w_v2t_denominator = exp_beta_sim.sum(dim=1, keepdim=True) - torch.diag_embed(torch.diag(exp_beta_sim))
#         w_v2t = (w_v2t_numerator / w_v2t_denominator) * weight_matrix  # 应用权重矩阵
        
#         # 计算分母权重 (video-to-text方向)
#         w_t2v_numerator = (batch_size - 1) * exp_beta_sim
#         w_t2v_denominator = exp_beta_sim.sum(dim=0, keepdim=True) - torch.diag_embed(torch.diag(exp_beta_sim))
#         w_t2v = (w_t2v_numerator / w_t2v_denominator) * weight_matrix.T  # 转置权重矩阵
        
#         # 保持对角线权重为alpha
#         diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
#         w_v2t[diag_mask] = self.alpha
#         w_t2v[diag_mask] = self.alpha
        
#         # ===== 最终损失计算 =====
#         # text-to-video损失
#         logits_v2t = torch.exp(sim_matrix) * w_v2t
#         log_prob_v2t = torch.log(logits_v2t.sum(dim=1)) - torch.diag(sim_matrix)
        
#         # video-to-text损失
#         logits_t2v = torch.exp(sim_matrix) * w_t2v
#         log_prob_t2v = torch.log(logits_t2v.sum(dim=0)) - torch.diag(sim_matrix)
        
#         # 总损失
#         loss = (log_prob_v2t.mean() + log_prob_t2v.mean()) / 2
        
#         return loss

class CrossModalHardNegativeNCE(nn.Module):
    def __init__(self, 
                 alpha: float = 1.0, 
                 beta: float = 0.0,
                 tau_vis: float = 0.7,    # ⭐视觉相似度阈值
                 tau_txt: float = 0.3,    # ⭐文本相似度阈值
                 gamma: float = 1.5,     # ⭐跨模态惩罚系数
                 **kwargs):
        super(CrossModalHardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau_vis = tau_vis
        self.tau_txt = tau_txt
        self.gamma = gamma

        self.hardNegativeNCE = HardNegativeNCE(alpha, 0.5)

    def forward(
        self,
        vis_sim_matrix: torch.Tensor,  # ⭐重命名为视觉相似度矩阵
        txt_sim_matrix: torch.Tensor,  # ⭐新增文本相似度矩阵
        video_embds: torch.Tensor,
        text_embds: torch.Tensor,
        temp,
    ):
        """
        Args:
            vis_sim_matrix: (batch_size, batch_size) 视频特征相似度
            txt_sim_matrix: (batch_size, batch_size) 标题文本相似度
        """
        batch_size = vis_sim_matrix.size(0)
        
        # ========== 原始NCE计算流程 ==========
        sim_matrix = vis_sim_matrix / temp
        sim_matrix = sim_matrix.float()
        
        nominator = torch.diagonal(sim_matrix)
        beta_sim = self.beta * sim_matrix
        
        # 计算原始权重
        w_v2t = ((batch_size - 1) * torch.exp(beta_sim) / 
                (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim))))
        w_t2v = ((batch_size - 1) * torch.exp(beta_sim) / 
                (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim))))

        # ========== ⭐跨模态硬负样本挖掘 ==========
        with torch.no_grad():
            # 生成跨模态掩码 (视觉相似但文本不相似)
            hard_neg_mask = (vis_sim_matrix > self.tau_vis) & \
                           (txt_sim_matrix < self.tau_txt)
            
            # 排除对角线（正样本对）
            hard_neg_mask.fill_diagonal_(False)

        # ========== ⭐权重增强策略 ==========
        # 对伪困难样本进行权重惩罚
        w_v2t[hard_neg_mask] *= self.gamma  
        w_t2v[hard_neg_mask] *= self.gamma
        
        # 保持正样本权重不变
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        # ========== 最终损失计算 ==========
        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))
        
        loss = (denominator_v2t - nominator).mean() + \
               (denominator_t2v - nominator).mean()
               
        # ========== LOSS_2 的损失 ==========
        # 计算 LOSS_2 的损失
        loss_2 = self.hardNegativeNCE(video_embds, text_embds, temp)
        loss = loss + loss_2*0.3

        # ⭐可选：返回掩码统计信息用于监控
        aux_info = {
            'hard_neg_ratio': hard_neg_mask.float().mean().item(),
            'avg_vis_sim': vis_sim_matrix[hard_neg_mask].mean().item() if hard_neg_mask.any() else 0,
            'avg_txt_sim': txt_sim_matrix[hard_neg_mask].mean().item() if hard_neg_mask.any() else 0
        }
        
        return loss, aux_info
