import torch
import os
#改动
#因为这里有些pth文件加载不出来，报错了,所以我要调试一下
def load_target_embedding(target_pth):
    try:
        # 尝试加载文件
        target_emb = torch.load(target_pth, weights_only=True).cpu().to(torch.float32)
        return target_emb
    except Exception as e:
        # 如果发生异常，打印错误信息和文件路径
        print(f"加载PTH文件失败，文件名为: {target_pth}")
        print(f"错误日志: {e}")
        # 可以选择在这里记录日志或者进行其他处理
        # 例如：logging.error(f"Failed to load file: {target_pth}. Error: {e}")
        # 如果需要，可以重新抛出异常
        return None


# 自定义collate_fn
def collate_fn(batch):
    # 过滤掉所有无效样本（None）
    batch = [item for item in batch if item is not None]

    # 如果过滤后批次为空，返回 None（需配合 DataLoader 的 drop_last=True）
    if len(batch) == 0:
        return None

    # 将有效样本组合成张量
    return torch.utils.data.dataloader.default_collate(batch)

#调试哪些文件损坏了
if __name__ == "__main__":
    folder_path = "/stu2024new/ygy/datasets/WebVid/2M/blip2-vid-embs-large-all/3"
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pth"):
            file_path = os.path.join(folder_path, file_name)
            try:
                torch.load(file_path, map_location=torch.device('cpu'))
                # print(f"File {file_name} is valid.")
            except Exception as e:
                print(f"File {file_name} is corrupted: {e}")