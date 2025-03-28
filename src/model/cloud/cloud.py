from utils import getEX_EN_HE_dim_direction, reparameterize
import torch

class Cloud:
    def __init__(self , x:torch.Tensor , dim:int , num:int):
        self.x = x
        self.dim = dim
        self.num = num
        ex , en ,he = getEX_EN_HE_dim_direction(self.x , self.dim , self.num)
        self.ex = ex
        self.en = en
        self.he = he
        new_x , enn = reparameterize(ex,en,he)
        self.new_x = new_x
        self.enn = enn

    def get_cloud(self) -> torch.Tensor:
        return self.new_x

    def get_ex_en_he_dim_direction(self) -> tuple:
        return self.ex, self.en, self.he

    def get_enn(self) -> torch.Tensor:
        return self.enn


if __name__ == "__main__":
    x = torch.randn(32, 512)
    cloud = Cloud(x, 0)
    x = cloud.get_cloud()
    print(f'x.shape: {x.shape}')