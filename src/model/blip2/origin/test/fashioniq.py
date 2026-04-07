import datetime
import time
from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabulate import tabulate

from src.model.blip2.xpool_cross_att import Transformer as xpool_cross_att
from src.model.blip2.xpool_cross_att import sim_matrix_training
from src.tools.files import json_dump, json_load
import gc #做垃圾回收, 内存不够,跑不起来
from pathlib import Path
import os #清理磁盘文件 permute,sqeeze

class TestFashionIQ:
    def __init__(self, category: str):
        self.category = category
        pass

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation...")
        start_time = time.time()

        query_feats = []
        captions = []
        idxs = []
        #特化数据库的指导信息 todo
        # guide_qs = []
        # query_feats_cross = []
        for batch in data_loader:
            ref_img = batch["ref_img"]
            caption = batch["edit"]
            idx = batch["pair_id"]

            idxs.extend(idx.cpu().numpy().tolist())
            captions.extend(caption)

            device = ref_img.device

            ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))
            ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
                device
            )

            text_tokens = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            query_tokens = model.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                device
            )
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            query_embs = model.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=ref_img_atts,
                return_dict=True,
            )
            query_feat = query_embs.last_hidden_state[:, : query_tokens.size(1), :]
            query_feat = F.normalize(model.text_proj(query_feat), dim=-1)

            
            # 改动5
            #-------------------------------------------------------------------------------------
            # query_si_feats = vl_feat #我保留一下mean之前的结果
            # tar_feat = tar_feat.mean(dim=1)
            query_feat = query_feat.mean(dim=1)

            #开始你的表演---------------------------------
            # query_si_feat = query_feat

            # edit_query_embs = model.Qformer.bert(
            #     text_tokens.input_ids,
            #     attention_mask=text_tokens.attention_mask,
            #     return_dict=True,
            # ) 
            
            # edit_query_embs = edit_query_embs.last_hidden_state[:, :, :]# [bs, 32, 768]
            # ca_img = model.cat_proj_img(ref_img_embs) #(B,677,256)
            # ca_img = F.normalize(ca_img, dim=-1)
            # ca_edit = model.cat_proj_edit(edit_query_embs) #(B,32,256)
            # ca_edit = F.normalize(ca_edit, dim=-1)

            # guide_q = model.ca(ca_img , ca_edit)
            
            # guide_qs.append(guide_q.cpu())
            
            # edit_query_embs = model.Qformer.bert(
            #     text_tokens.input_ids,
            #     attention_mask=text_tokens.attention_mask,
            #     return_dict=True,
            # ) # [bs, 32, 768]
        
            # edit_vl_embs = edit_query_embs.last_hidden_state[:, 0, :] # 取 [CLS] token #全局特征 [bs, 768]

            # orig_query_embs = orig_query_embs.last_hidden_state[:, :, :] # 取全部tensor,作为一个序列,保留信息 [bs, 32, 768]

            # ==========原视频信息 todo=============
            # ref_img_embs_cls = ref_img_embs[:, 0, :] #(bs , 1048)
            # ref_img_embs_cls = model.ref_img_proj(ref_img_embs_cls)#(bs,1024)
            # =================================

            #cat
            # combined = torch.cat([query_si_feat, edit_vl_embs], dim=1)
            # combined = torch.cat([combined, ref_img_embs_cls], dim=1) #1024+1024=2048
            # combined = model.cat_proj_0(combined) #1024
            # combined = model.cat_proj_1(combined)#只是用作特化数据库的指导信息，并不是真正的查询
            # query_si_feat_cross = model.cat_proj_2(combined)
            # query_feats_cross.append(query_si_feat_cross.cpu())

            # query_feat = query_si_feat
            #-------------------------------------------------------------------------

            #---------------------------------------------------------------------------------------


            query_feats.append(query_feat.cpu())

        query_feats = torch.cat(query_feats, dim=0)
        query_feats = F.normalize(query_feats, dim=-1)
        # guide_qs = torch.cat(guide_qs, dim=0)
        # guide_qs = F.normalize(guide_qs, dim=-1)
        # query_feats_cross = F.normalize(query_feats_cross, dim=-1) #指导信息也要normal
        # query_feats = query_feats.mean(dim=1) 我在过程里面mean了
        idxs = torch.tensor(idxs, dtype=torch.long)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            idxs = fabric.all_gather(idxs)

            query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
            idxs = einops.rearrange(idxs, "d b -> (d b)")

        if fabric.global_rank == 0:
            idxs = idxs.cpu().numpy()
            ref_img_ids = [data_loader.dataset.pairid2ref[idx] for idx in idxs]
            ref_img_ids = [data_loader.dataset.int2id[id] for id in ref_img_ids]

            tar_img_feats = []
            tar_img_ids = []
            for target_id in data_loader.dataset.target_ids:
                tar_img_ids.append(target_id)
                target_emb_pth = data_loader.dataset.id2embpth[target_id]
                target_feat = torch.load(target_emb_pth, weights_only=True).cpu()
                tar_img_feats.append(target_feat.cpu())
            tar_img_feats = torch.stack(tar_img_feats)
            
            tar_img_feats = tar_img_feats.to(query_feats.device)

            tar_img_feats = tar_img_feats.mean(dim=1)
            tar_img_feats = F.normalize(tar_img_feats, dim=-1)

            #================磁盘测评=======================
            # 1. 在移动硬盘上创建内存映射文件（注意：使用绝对路径！）
            # print(f'准备初始化块空间')
            # mmap_path = Path("/media/bd/PSSD1/cross_feats.mmap")  # ← 修改为您的实际路径
            # cross_feats_shape = (query_feats.size(0), tar_img_feats.size(0),256)  # 最终矩阵形状
            # # 初始化内存映射文件（提前分配空间）
            # cross_feats = np.memmap(mmap_path, 
            #                     dtype='float32',  # 节省空间
            #                     mode='w+',        # 读写模式
            #                     shape=cross_feats_shape)   # 预分配空间
            # print(f'初始化块空间成功')
            # # 分批计算交叉注意力
            # original_device = next(model.xpool_cross_att.parameters()).device
            # xpool_cross_att = model.xpool_cross_att
            # xpool_cross_att.cpu()
            # batch_size = 100  # 根据GPU内存调整
            # print('no guide_qs but query_feats')
            # x = query_feats.size(0)
            # for i in range(0, x, batch_size):
            #     batch_q = query_feats[i:i+batch_size]
            #     print(f'batch_q的形状:{batch_q.shape}')
            #     print(f'tar_img_feats的形状:{tar_img_feats.shape}')
            #     # 关键修改：直接计算并写入磁盘
            #     cross_feats[i:i+batch_size] = xpool_cross_att(batch_q, tar_img_feats).permute(1,0,2).cpu().numpy()
            #     del batch_q
            #     gc.collect()
            #     if i % 100 == 0: 
            #         print(f'Processing batch {i}/{x}')
            # print(f'cross_feats应该为(B,B,256):{cross_feats.shape}')
            # model.xpool_cross_att.to(original_device)  # 立即恢复原设备
            # gc.collect()

            #=====================================================
            batch_size_q = 200  # 查询分批大小（根据GPU内存调整）
            batch_size_t = 200  # 目标视频分批大小
            device = 'cuda'
            sims_q2t_t = np.zeros((query_feats.size(0), tar_img_feats.size(0)))  # 预分配最终结果矩阵
            for i in range(0, query_feats.size(0), batch_size_q):
                # 分批加载查询特征（CPU → GPU）
                vl_batch = query_feats[i:i+batch_size_q].to(device)  # (100, 256)
                
                for j in range(0, tar_img_feats.size(0), batch_size_t):
                    # 分批加载交叉特征（直接从磁盘映射）
                    # cross_batch = torch.from_numpy(
                    #     cross_feats[i:i+batch_size_q, j:j+batch_size_t]
                    # ).permute(1,0,2).to(device)  # (100, 100, 256)
                    tar_batch = tar_img_feats[j:j+batch_size_t]
                    tar_batch = tar_batch.to(device)
                    # 计算局部相似度
                    sim_batch = torch.einsum("ie,je->ij", vl_batch, tar_batch)
                    # 计算局部相似度
                    # sim_batch = sim_matrix_training(vl_batch, cross_batch, 'max')  # (100, 100)
                    
                    # 填充到最终矩阵
                    sims_q2t_t[i:i+batch_size_q, j:j+batch_size_t] = sim_batch.cpu().numpy()
                    
                    # del cross_batch, sim_batch
                    # torch.cuda.empty_cache()
                    # gc.collect()

            print(f'numpy转化成tensor中')
            sim_q2t = torch.from_numpy(sims_q2t_t)
            print(f'转化tensor成功')
            # os.remove('/media/bd/PSSD1/cross_feats.mmap') 
            # del sims_q2t_t
            # gc.collect()
            print(f'sims_q2t的形状：{sim_q2t.shape}')

            #===============================================
            #=============内存测评,计算特化数据库，生成相似矩阵=====================
            # #计算xpool_hn_nce需要的对比矩阵
            # print("计算xpool_hn_nce需要的对比矩阵")
            # # tar_img_feat_add_frame = tar_img_feats.unsqueeze(1) #变成(B , 1 ,dim)
            # original_device = next(model.xpool_cross_att.parameters()).device
            # xpool_cross_att = model.xpool_cross_att
            # xpool_cross_att.cpu()
            # cross_feats = xpool_cross_att(query_feats , tar_img_feats) #交叉结果(B,B,dim)
            # model.xpool_cross_att.to(original_device)  # 立即恢复原设备

            # xpool_sim_matrix = sim_matrix_training(query_feats , cross_feats,'max') #(B,B)对比矩阵
            # sim_q2t = xpool_sim_matrix.cpu()
            #=========================================================

            # sim_q2t = (query_feats @ tar_img_feats.t()).cpu() #不用普通的相似矩阵

            # Add zeros where ref_img_id == tar_img_id
            for i in range(len(ref_img_ids)):
                for j in range(len(tar_img_ids)):
                    if ref_img_ids[i] == tar_img_ids[j]:
                        sim_q2t[i][j] = -10

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            ref_img_ids = np.array(ref_img_ids)
            tar_img_ids = np.array(tar_img_ids)

            cor_img_ids = [data_loader.dataset.pairid2tar[idx] for idx in idxs]
            cor_img_ids = [data_loader.dataset.int2id[id] for id in cor_img_ids]

            recalls = get_recalls_labels(sim_q2t, cor_img_ids, tar_img_ids)
            fabric.print(recalls)

            # Save results
            json_dump(recalls, f"recalls_fiq-{self.category}.json")

            print(f"Recalls saved in {Path.cwd()}/recalls_fiq-{self.category}.json")

            mean_results(fabric=fabric)

        fabric.barrier()


# From google-research/composed_image_retrieval
def recall_at_k_labels(sim, query_lbls, target_lbls, k=10):
    distances = 1 - sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_lbls)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(query_lbls), len(target_lbls)).reshape(
            len(query_lbls), -1
        )
    )
    assert torch.equal(
        torch.sum(labels, dim=-1).int(), torch.ones(len(query_lbls)).int()
    )
    return round((torch.sum(labels[:, :k]) / len(labels)).item() * 100, 2)


def get_recalls_labels(
    sims, query_lbls, target_lbls, ks: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    return {f"R{k}": recall_at_k_labels(sims, query_lbls, target_lbls, k) for k in ks}


def mean_results(dir=".", fabric=None, save=True):
    dir = Path(dir)
    recall_pths = list(dir.glob("recalls_fiq-*.json"))
    recall_pths.sort()
    if len(recall_pths) != 3:
        return

    df = {}
    for pth in recall_pths:
        name = pth.name.split("_")[1].split(".")[0]
        data = json_load(pth)
        df[name] = data

    df = pd.DataFrame(df)

    # FASHION-IQ
    df_fiq = df[df.columns[df.columns.str.contains("fiq")]]
    assert len(df_fiq.columns) == 3
    df_fiq["Average"] = df_fiq.mean(axis=1)
    df_fiq["Average"] = df_fiq["Average"].apply(lambda x: round(x, 2))

    headers = [
        "dress\nR10",
        "dress\nR50",
        "shirt\nR10",
        "shirt\nR50",
        "toptee\nR10",
        "toptee\nR50",
        "Average\nR10",
        "Average\nR50",
    ]
    fiq = []
    for category in ["fiq-dress", "fiq-shirt", "fiq-toptee", "Average"]:
        for recall in ["R10", "R50"]:
            value = df_fiq.loc[recall, category]
            value = str(value).zfill(2)
            fiq.extend([value])
    if fabric is None:
        print(tabulate([fiq], headers=headers, tablefmt="latex_raw"))
        print(" & ".join(fiq))
    else:
        fabric.print(tabulate([fiq], headers=headers))
        fabric.print(" & ".join(fiq))

    if save:
        df_mean = df_fiq["Average"].to_dict()
        df_mean = {k + "_mean": round(v, 2) for k, v in df_mean.items()}
        json_dump(df_mean, "recalls_fiq-mean.json")
