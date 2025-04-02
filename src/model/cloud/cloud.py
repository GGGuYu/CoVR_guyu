from src.model.cloud.utils import getEX_EN_HE_dim_direction, reparameterize
import torch

class Cloud:
    def __init__(self , x:torch.Tensor , dim:int , num:int):
        self.eps = 1e-6
        self.x = x
        self.dim = dim
        self.num = num
        ex , en ,he = getEX_EN_HE_dim_direction(self.x , self.dim , self.num)
        self.ex = ex
        self.en = en
        self.he = he
        new_x , enn , mu = reparameterize(ex,en,he)
        # print(f'new_x重参数化之后:{new_x.shape}')
        x_std = torch.sqrt(torch.var(x, dim=-1, keepdim=False) + self.eps)
        # print(f'x_std:{x_std.shape}')
        normal_x = (x - ex) / x_std.unsqueeze(dim=1) #做点改动
        # print(f'normal_x:{normal_x.shape}')
        new_x = enn * normal_x + new_x
        # print(f'new_x第一次不确定加噪:{new_x.shape}')
        sigma = torch.rand_like(x)
        # print(f'sigma:{sigma.shape}')
        self.new_x = new_x + sigma * mu
        # print(f'new_x第二次不确定加噪:{self.new_x.shape}')
        self.enn = enn

    def get_cloud(self) -> torch.Tensor:
        return self.new_x

    def get_ex_en_he_dim_direction(self) -> tuple:
        return self.ex, self.en, self.he

    def get_enn(self) -> torch.Tensor:
        return self.enn


if __name__ == "__main__":
    # 测试1: 基本功能测试
    print("=== 测试1: 基本功能 ===")
    x = torch.randn(32, 512)
    cloud = Cloud(x, 1, x.shape[1])
    cloud_output = cloud.get_cloud()
    print(f"输入形状: {x.shape}, 输出形状: {cloud_output.shape}")
    print(f"输入均值: {x.mean():.4f}, 输出均值: {cloud_output.mean():.4f}")
    print(f"输入标准差: {x.std():.4f}, 输出标准差: {cloud_output.std():.4f}\n")

    # 测试2: 设备一致性测试
    print("=== 测试2: 设备一致性 ===")
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        cloud_cuda = Cloud(x_cuda, 1, x_cuda.shape[1])
        print(f"CPU输出设备: {cloud_output.device}")
        print(f"GPU输出设备: {cloud_cuda.get_cloud().device}\n")
    else:
        print("未检测到CUDA设备\n")

    # 测试3: 参数获取测试
    print("=== 测试3: 参数获取 ===")
    ex, en, he = cloud.get_ex_en_he_dim_direction()
    enn = cloud.get_enn()
    print(f"Ex形状: {ex.shape}, 值范围: [{ex.min():.4f}, {ex.max():.4f}]")
    print(f"En形状: {en.shape}, 值范围: [{en.min():.4f}, {en.max():.4f}]")
    print(f"He形状: {he.shape}, 值范围: [{he.min():.4f}, {he.max():.4f}]")
    print(f"Enn形状: {enn.shape}, 值范围: [{enn.min():.4f}, {enn.max():.4f}]")