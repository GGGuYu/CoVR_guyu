import torch
import math


def reparameterize(ex, en, he):
    enn_0 = torch.exp(he / 2)
    enn_norm = torch.randn_like(enn_0)  # 返回一个与输入相同大小的张量，该张量由均值为0、方差为1的正态分布的随机数填充。
    enn = en + enn_norm * enn_0

    enn_1 = torch.exp(enn / 2)
    x_norm = torch.randn_like(enn_1)  # 返回一个与输入相同大小的张量，该张量由均值为0、方差为1的正态分布的随机数填充。
    x = ex + x_norm * enn_1

    return x, enn


def getEX_EN_HE_dim_direction(x, dim, num):
    """
    沿批次维度（第0维）计算每个特征维度的云参数
    输入: x (Tensor) - 形状 (batch_size, embedding_dim) 例如 (32, 512)
    返回: (Ex, En, He) - 每个形状 (embedding_dim,)
    """
    # 计算期望 Ex = 均值（沿批次方向）
    ex = torch.mean(x, dim=dim)  # (512,)

    # 计算熵 En = (sqrt(π/2)/2) * 平均绝对偏差
    abs_dev = torch.abs(x - ex.unsqueeze(dim))  # 绝对偏差 (32, 512)
    mean_abs_dev = torch.mean(abs_dev, dim=dim)  # (512,)
    en = (math.sqrt(math.pi / 2)) * mean_abs_dev  # (512,)

    # 计算超熵 He = sqrt(样本方差 - En²)
    sample_var = torch.var(x, dim=dim, unbiased=True)  # 无偏方差 (512,)
    he_squared = sample_var - torch.square(en)
    he = torch.sqrt(torch.clamp(he_squared, min=1e-6))  # 防止负值

    ex, en, he = expand_dim(ex, en, he, dim, num)

    return ex, en, he


def expand_dim(ex, en, he, dim, num):
    return expand_sigle(ex, dim, num), expand_sigle(en, dim, num), expand_sigle(he, dim, num)


def expand_single(x, dim, num):
    """
    在指定维度上将张量扩展指定的份数。

    参数:
        x (torch.Tensor): 输入的1维张量
        dim (int): 需要扩展的维度
        num (int): 扩展的份数

    返回:
        torch.Tensor: 扩展后的张量
    """
    # 确保输入是一个1维张量
    if x.dim() != 1:
        raise ValueError("输入张量必须是1维的")

    # 在指定维度上插入一个新维度
    x_expanded = x.unsqueeze(dim)

    # 构造扩展的形状
    expand_shape = list(x_expanded.shape)
    expand_shape[dim] = num

    # 扩展张量
    x_expanded = x_expanded.expand(expand_shape)

    return x_expanded


if __name__ == "__main__":
    x = torch.randn(32, 512)
    num = x.shape[0]
    ex, en, he = getEX_EN_HE_dim_direction(x, 0, num)
    print(f"ex.shape: {ex.shape},\n en.shape: {en.shape},\n he.shape: {he.shape}")

    x, enn = reparameterize(ex, en, he)
    print(f'x.shape: {x.shape}')