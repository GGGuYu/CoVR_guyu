import torch
import math

#正向云
def reparameterize(ex, en, he):
    # enn_0 = torch.exp(he / 2)
    enn_0 = he
    enn_norm = torch.randn_like(enn_0)  # 返回一个与输入相同大小的张量，该张量由均值为0、方差为1的正态分布的随机数填充。
    enn = en + enn_norm * enn_0

    # enn_1 = torch.exp(enn / 2)
    enn_1 = enn
    x_norm = torch.randn_like(enn_1)  # 返回一个与输入相同大小的张量，该张量由均值为0、方差为1的正态分布的随机数填充。
    x = ex + x_norm * enn_1

    # 计算隶属度 μ = exp(-(x - Ex)^2 / (2 * En^2))
    mu = torch.exp(-torch.pow(x - ex, 2) / (2 * torch.pow(enn, 2)))

    return x, enn, mu  # 现在返回三个值：x, enn和隶属度mu

#逆向云
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
    return expand_single(ex, dim, num), expand_single(en, dim, num), expand_single(he, dim, num)

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
    print("=== 测试1: expand_single基本功能 ===")
    test_tensor = torch.randn(512)
    expanded = expand_single(test_tensor, dim=0, num=32)
    print(f"输入形状: {test_tensor.shape}, 输出形状: {expanded.shape}")
    print(f"输入值范围: [{test_tensor.min():.4f}, {test_tensor.max():.4f}]")
    print(f"输出值范围: [{expanded.min():.4f}, {expanded.max():.4f}]\n")

    print("=== 测试2: getEX_EN_HE_dim_direction功能 ===")
    x = torch.randn(32, 512)
    ex, en, he = getEX_EN_HE_dim_direction(x, dim=0, num=x.shape[0])
    print(f"Ex形状: {ex.shape}, 值范围: [{ex.min():.4f}, {ex.max():.4f}]")
    print(f"En形状: {en.shape}, 值范围: [{en.min():.4f}, {en.max():.4f}]")
    print(f"He形状: {he.shape}, 值范围: [{he.min():.4f}, {he.max():.4f}]\n")

    print("=== 测试3: reparameterize功能 ===")
    test_ex = torch.randn(32, 512)
    test_en = torch.rand(32, 512)
    test_he = torch.rand(32, 512)
    x, enn, mu = reparameterize(test_ex, test_en, test_he)
    print(f"输出x形状: {x.shape}, 值范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"输出enn形状: {enn.shape}, 值范围: [{enn.min():.4f}, {enn.max():.4f}]")
    print(f"输出mu形状: {mu.shape}, 值范围: [{mu.min():.4f}, {mu.max():.4f}]\n")

    print("=== 测试4: 设备一致性测试 ===")
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        ex_cuda, en_cuda, he_cuda = getEX_EN_HE_dim_direction(x_cuda, dim=0, num=x_cuda.shape[0])
        print(f"CPU输出设备: {x.device}")
        print(f"GPU输出设备: {ex_cuda.device}")
    else:
        print("未检测到CUDA设备")