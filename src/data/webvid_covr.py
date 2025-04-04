import ast
import random
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import transform_test, transform_train
from src.data.utils import FrameLoader, id2int, pre_caption
from src.tools.files import write_txt
from src.tools.utils import print_dist
from src.data.my_utils import load_target_embedding
from src.data.my_utils import collate_fn

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class WebVidCoVRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        vid_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        emb_pool: str = "query",
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        n_embs: int = 15,
        si_tc_weight=0,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)

        self.data_train = WebVidCoVRDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            vid_dir=vid_dirs["train"],
            emb_dir=emb_dirs["train"],
            split="train",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            n_embs=n_embs,
            si_tc_weight=si_tc_weight,
        )
        self.data_val = WebVidCoVRDataset(
            transform=self.transform_test,
            annotation=annotation["val"],
            vid_dir=vid_dirs["val"],
            emb_dir=emb_dirs["val"],
            split="val",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            n_embs=n_embs,
        )

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class WebVidCoVRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        vid_dirs: str,
        emb_dirs: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        emb_pool: str = "query",
        n_embs: int = 15,
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames

        self.transform_test = transform_test(image_size)

        self.data_test = WebVidCoVRDataset(
            transform=self.transform_test,
            annotation=annotation,
            vid_dir=vid_dirs,
            emb_dir=emb_dirs,
            split="test",
            emb_pool=self.emb_pool,
            n_embs=n_embs,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class WebVidCoVRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        vid_dir: str,
        emb_dir: str,
        split: str,
        max_words: int = 30,
        emb_pool: str = "query",
        n_embs: int = 15,
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        si_tc_weight=0,
    ) -> None:
        super().__init__()

        self.transform = transform

        self.annotation_pth = Path(annotation)
        assert (
            self.annotation_pth.exists()
        ), f"Annotation file {annotation} does not exist"
        self.df = pd.read_csv(annotation)

        self.vid_dir = Path(vid_dir)
        self.emb_dir = Path(emb_dir)
        assert self.vid_dir.exists(), f"Image directory {self.vid_dir} does not exist"
        assert self.emb_dir.exists(), f"Embedding directory {emb_dir} does not exist"

        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        self.split = split

        vid_pths = self.vid_dir.glob("*/*.mp4")
        emb_pths = self.emb_dir.glob("*/*.pth")

        id2vidpth = {
            vid_pth.parent.stem + "/" + vid_pth.stem: vid_pth for vid_pth in vid_pths
        }
        id2embpth = {
            emb_pth.parent.stem + "/" + emb_pth.stem: emb_pth for emb_pth in emb_pths
        }

        assert len(id2vidpth) > 0, f"No videos found in {vid_dir}"
        assert len(id2embpth) > 0, f"No embeddings found in {emb_dir}"


        # 改动
        # 我先手动跳过缺失的条目------------------------------------------------------
        vid_pths = self.vid_dir.glob("*/*.mp4")
        # Ensure id2vidpth is not empty
        assert len(id2vidpth) > 0, f"No video paths found in {self.vid_dir}"
        # Safely apply mapping to create path1
        self.df["path1"] = self.df["pth1"].apply(lambda x: id2vidpth.get(x, None))
        # Filter and reset index for missing paths
        self.df = self.df[self.df["path1"].notna()].reset_index(drop=True)
        # Create path2 mapping
        emb_pths = self.emb_dir.glob("*/*.pth")
        id2embpth = {emb_pth.parent.stem + "/" + emb_pth.stem: emb_pth for emb_pth in emb_pths}
        # Safely apply mapping to create path2
        self.df["path2"] = self.df["pth2"].apply(lambda x: id2embpth.get(x, None))
        # Filter and reset index for missing path2 entries
        self.df = self.df[self.df["path2"].notna()].reset_index(drop=True)

        #------------------------------------------------------------------------------------


        self.df["path1"] = self.df["pth1"].apply(lambda x: id2vidpth.get(x, None))  # type: ignore
        self.df["path2"] = self.df["pth2"].apply(lambda x: id2embpth.get(x, None))  # type: ignore

        # Count unique missing paths
        missing_pth1 = self.df[self.df["path1"].isna()]["pth1"].unique().tolist()
        missing_pth1.sort()
        total_pth1 = self.df["pth1"].nunique()

        missing_pth2 = self.df[self.df["path2"].isna()]["pth2"].unique().tolist()
        missing_pth2.sort()
        total_pth2 = self.df["pth2"].nunique()

        if len(missing_pth1) > 0:
            print_dist(
                f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%), saving them to missing_pth1-{split}.txt"
            )
            if split == "test":
                raise ValueError(
                    f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%) in test split"
                )
            write_txt(missing_pth1, f"missing_pth1-{split}.txt")
        if len(missing_pth2) > 0:
            print_dist(
                f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%), saving them to missing_pth2-{split}.txt"
            )
            if split == "test":
                raise ValueError(
                    f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%) in test split"
                )
            write_txt(missing_pth2, f"missing_pth2-{split}.txt")

        # Remove missing paths
        self.df = self.df[self.df["path1"].notna()]
        self.df = self.df[self.df["path2"].notna()]
        self.df.reset_index(drop=True, inplace=True)

        self.max_words = max_words

        assert emb_pool in [
            "middle",
            "mean",
            "query",
        ], f"Invalid emb_pool: {emb_pool}, must be one of middle, mean, or query"
        self.emb_pool = emb_pool
        self.n_embs = n_embs

        if iterate in ["idx", "triplets"]:
            iterate = "idx"
            self.df["idx"] = self.df.index
        self.iterate = iterate
        self.target_txts = self.df[iterate].unique()
        assert (
            iterate in self.df.columns
        ), f"{iterate} not in {self.annotation_pth.stem}"
        self.df.sort_values(iterate, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df["int1"] = self.df["pth1"].apply(lambda x: id2int(x, sub="0"))
        self.df["int2"] = self.df["pth2"].apply(lambda x: id2int(x, sub="0"))
        self.pairid2ref = self.df["int1"].to_dict()
        assert (
            self.df["int1"].nunique() == self.df["pth1"].nunique()
        ), "int1 is not unique"
        assert (
            self.df["int2"].nunique() == self.df["pth2"].nunique()
        ), "int2 is not unique"
        # int2id is a dict with key: int1, value: pth1
        self.int2id = self.df.groupby("int1")["pth1"].apply(set).to_dict()
        self.int2id = {k: list(v)[0] for k, v in self.int2id.items()}

        self.pairid2tar = self.df["int2"].to_dict()
        self.df.set_index(iterate, inplace=True)
        self.df[iterate] = self.df.index

        if split == "test":
            assert (
                len(self.target_txts) == self.df.shape[0]
            ), "Test split should have one caption per row"

        assert (
            vid_query_method
            in [
                "middle",
                "random",
                "sample",
            ]
        ), f"Invalid vid_query_method: {vid_query_method}, must be one of middle, random, or sample"
        self.frame_loader = FrameLoader(
            transform=self.transform, method=vid_query_method, frames_video=vid_frames
        )

        # Load text embeddings if si_tc_weight > 0
        self.txt2emb = None
        if si_tc_weight > 0:
            txt2emb_pth = self.emb_dir / f"txt2_{self.annotation_pth.stem}.pth"
            if "blip2" in str(txt2emb_pth):
                model = "blip2"
            elif "blip" in str(txt2emb_pth):
                model = "blip"
            elif "clip" in str(txt2emb_pth):
                model = "clip"
            else:
                raise ValueError(f"Invalid model: {txt2emb_pth}")
            assert txt2emb_pth.exists(), f"txt2emb does not exist: {txt2emb_pth}. Please compute them with: python tools/embs/save_{model}_embs_txts.py {self.annotation_pth} {self.emb_dir}"
            txt2emb_pth = self.emb_dir / f"txt2_{self.annotation_pth.stem}.pth"
            if txt2emb_pth.exists():
                self.txt2emb = torch.load(txt2emb_pth, weights_only=True)
                assert len(self.txt2emb["texts"]) == len(
                    self.txt2emb["feats"]
                ), "txt2emb is not valid"
                self.txt2emb = {
                    txt: feat
                    for txt, feat in zip(self.txt2emb["texts"], self.txt2emb["feats"])
                }
                txt2s = set(self.df["txt2"].unique().tolist())
                assert txt2s.issubset(
                    set(self.txt2emb.keys())
                ), "txt2emb does not contain all txt2's"

    def __len__(self) -> int:
        return len(self.target_txts)

    def __getitem__(self, index):
        target_txt = self.target_txts[index]
        ann = self.df.loc[target_txt]
        if ann.ndim > 1:
            ann = ann.sample()
            ann = ann.iloc[0]

        reference_pth = str(ann["path1"])
        reference_vid = self.frame_loader(reference_pth)

        caption = pre_caption(ann["edit"], self.max_words)

        return_dict = {
            "ref_img": reference_vid,
            "edit": caption,
            "pair_id": index,
        }

        if self.txt2emb is not None:
            return_dict["tar_txt_feat"] = self.txt2emb[ann["txt2"]]

        # Get target embeddings
        target_pth = str(ann["path2"])
        # target_emb = torch.load(target_pth, weights_only=True).cpu().to(torch.float32)
        #-----------------------------------------------------------
        #改动，替换成我的加载方式，方便调试
        # target_emb = torch.load(target_pth, weights_only=True).cpu().to(torch.float32)
        target_emb = load_target_embedding(target_pth)
        #我发现有极少数损坏文件，应该是我计算的时候几次中断导致的
        #但是应该只有个位数，并不影响实验结果，因此我打算设法先跳过这些样本
        #之后再来处理 todo
        if target_emb is None:
            # 如果加载失败，直接return None交给collate_fn函数去处理
            print(f"跳过了样本: {target_pth}")
            return None
        #我认为这样应该不会有大问题，因为只有个位数的损坏文件
        #为了解决这个问题我还关闭了pin_memory=False(补了一个配置在配置文件)
        #速度变慢了，但是似乎能跑了
        #DS建议我用的就是这个方法，以下是说明
        # 自定义 collate_fn
        # def collate_fn(batch):
        #     # 过滤无效样本
        #     batch = [item for item in batch if item is not None]
        #     if len(batch) == 0:
        #         return None
        #     # 默认的堆叠方法（适用于张量）
        #     return torch.utils.data.dataloader.default_collate(batch)
        #然后在初始化的时候传入
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=32,
        #     collate_fn=collate_fn,
        #     shuffle=True,
        #     drop_last=True,
        #     num_workers=4,
        # )
        #------------------------------------------------------------
        if self.emb_pool == "middle":
            return_dict["tar_img_feat"] = target_emb[len(target_emb) // 2]
            return return_dict

        n_target_emb = min(self.n_embs, len(target_emb))
        sampled_indices = random.sample(range(len(target_emb)), n_target_emb)
        sampled_target_emb = target_emb[sampled_indices]

        if self.emb_pool == "mean":
            return_dict["tar_img_feat"] = sampled_target_emb.mean(0)
            return return_dict

        assert self.emb_pool == "query", f"Invalid emb_pool: {self.emb_pool}"

        vid_scores = ast.literal_eval(str(ann["scores"]))
        if len(vid_scores) == 0 or len(target_emb) != len(vid_scores):
            vid_scores = [1.0] * n_target_emb
        else:
            vid_scores = [vid_scores[i] for i in sampled_indices]
        vid_scores = torch.Tensor(vid_scores)
        vid_scores = (vid_scores / 0.1).softmax(dim=0)
        if len(target_emb.shape) == 2:
            return_dict["tar_img_feat"] = torch.einsum(
                "f,fe->e", vid_scores, sampled_target_emb
            )
        elif len(target_emb.shape) == 3:
            return_dict["tar_img_feat"] = torch.einsum(
                "f,fqc->qc", vid_scores, sampled_target_emb
            )
        else:
            raise ValueError(
                f"target_emb must be 2 or 3 dimensional, got {len(target_emb.shape)}"
            )

        return return_dict
