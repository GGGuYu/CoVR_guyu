import datetime
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from src.model.blip2.xpool_cross_att import Transformer as xpool_cross_att
from src.model.blip2.xpool_cross_att import sim_matrix_training
from src.tools.files import json_dump
from src.tools.utils import concat_all_gather
import gc #做垃圾回收, 内存不够,跑不起来
from pathlib import Path
import os #清理磁盘文件 permute,sqeeze

class TestCirr:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for test...")
        start_time = time.time()

        vl_feats = []
        pair_ids = []
        #============特化数据库的指导信息 todo=============
        # query_feats_cross = []
        #===============================================
        for batch in data_loader:
            ref_img = batch["ref_img"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())

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

            output = model.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=ref_img_atts,
                return_dict=True,
            )

            vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
            vl_feat = F.normalize(model.text_proj(vl_embs), dim=-1)
            print(f'最开始vl_feat的形状：{vl_feat.shape}')

            # 改动5
            #-------------------------------------------------------------------------------------
            # query_si_feats = vl_feat #我保留一下mean之前的结果
            # tar_feat = tar_feat.mean(dim=1)
            vl_feat = vl_feat.mean(dim=1) #为什么他原来的没有mean呢，那不是32吗？todo 
            print(f'mean之后vl_feat的形状：{vl_feat.shape}')

            #开始你的表演---------------------------------
            query_si_feat = vl_feat
            
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

            vl_feat = query_si_feat
            #-------------------------------------------------------------------------

            #---------------------------------------------------------------------------------------


            # repeats = 32
            # 复制操作 (→ b, 32, 256)
            # expanded = vl_feat.unsqueeze(1)          # 先升维 (b,1,256)
            # vl_feat = expanded.repeat(1, repeats, 1)  # 沿第1维度复制32次



            vl_feats.append(vl_feat.cpu())

        pair_ids = torch.tensor(pair_ids, dtype=torch.long)
        vl_feats = torch.cat(vl_feats, dim=0)
        print(f'列表CAT后vl_feats的形状：{vl_feats.shape}') 

        #=======================normalize======================================
        # query_feats_cross = torch.cat(query_feats_cross, dim=0)
        vl_feats = F.normalize(vl_feats, dim=-1)
        # query_feats_cross = F.normalize(query_feats_cross, dim=-1)
        #=============================================================
        # vl_feats = concat_all_gather(vl_feats, fabric)
        # pair_ids = concat_all_gather(pair_ids, fabric)
        vl_feats = fabric.all_gather(vl_feats)
        pair_ids = fabric.all_gather(pair_ids)

        if fabric.global_rank == 0:
            pair_ids = pair_ids.cpu().numpy().tolist()

            assert len(vl_feats) == len(pair_ids)
            img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
            assert len(img_ids) == len(pair_ids)

            id2emb = OrderedDict()
            for img_id, target_emb_pth in data_loader.dataset.id2embpth.items():
                if img_id not in id2emb:
                    tar_emb = F.normalize(
                        torch.load(target_emb_pth, weights_only=True).cpu(), dim=-1
                    )
                    id2emb[img_id] = tar_emb

            tar_feats = torch.stack(list(id2emb.values()), dim=0).to("cpu")
            # vl_feats = vl_feats.to("cpu") #本来就在cpu上吧
            #=====================================================
            device = next(model.xpool_cross_att.parameters()).device
            # print(f'tar_feats的形状：{tar_feats.shape}')
            # tar_feats = tar_feats.mean(dim=1)
            tar_feats = F.normalize(tar_feats, dim=-1)
            print(f'tar_feats的形状：{tar_feats.shape}')
            #计算xpool_hn_nce需要的对比矩阵
            # tar_img_feat_add_frame = tar_feats
            print("计算xpool_hn_nce需要的对比矩阵")
            # tar_img_feat_add_frame = tar_feats.unsqueeze(1) #变成(B , 1 ,dim)
            # tar_feats = tar_feats.unsqueeze(1) #变成(B , 1 ,dim)
            # original_device = next(model.xpool_cross_att.parameters()).device
            # xpool_cross_att = model.xpool_cross_att
            # xpool_cross_att.cpu()
            # 1. 在移动硬盘上创建内存映射文件（注意：使用绝对路径！）
            print(f'准备初始化块空间')
            mmap_path = Path("/media/bd/PSSD/cross_feats.mmap")  # ← 修改为您的实际路径
            cross_feats_shape = (vl_feats.size(0), tar_feats.size(0),256)  # 最终矩阵形状
            # 初始化内存映射文件（提前分配空间）
            cross_feats = np.memmap(mmap_path, 
                                dtype='float32',  # 节省空间
                                mode='w+',        # 读写模式
                                shape=cross_feats_shape)   # 预分配空间
            print(f'初始化块空间成功')
            # 原一次性计算注释掉,cross_feats = xpool_cross_att(query_feats_cross , tar_feats).cpu().numpy()  #交叉结果(B,B,dim)
            # 分批计算交叉注意力
            # 数据放到gpu 一批一批
            
            tar_feats = tar_feats.to(device)
            batch_size = 128  # 根据GPU内存调整
            for i in range(0, vl_feats.size(0), batch_size):
                batch_q = vl_feats[i:i+batch_size]
                batch_q = batch_q.to(device)
                print(f'batch_q的形状:{batch_q.shape}') #(32,256)
                print(f'tar_feats的形状:{tar_feats.shape}') #(8082,32,256)
                # 关键修改：直接计算并写入磁盘
                cross_feats[i:i+batch_size] = model.xpool_cross_att(batch_q, tar_feats).permute(1,0,2).cpu().numpy()
                del batch_q
                torch.cuda.empty_cache()
                gc.collect()
                if i % 128 == 0: 
                    print(f'Processing batch {i}/{len(vl_feats)}')
            print(f'cross_feats应该为(B1,B2,256):{cross_feats.shape}')
            # model.xpool_cross_att.to(original_device)  # 立即恢复原设备
            # tar_feats = tar_feats.squeeze(1) #变成(B ,dim)
            # print(f'tar_feats最后的形状:{tar_feats.shape}')
            y = tar_feats.size(0)
            del tar_feats
            gc.collect()
            #=====================================================
            batch_size_q = 100  # 查询分批大小（根据GPU内存调整）
            batch_size_t = 100  # 目标视频分批大小
            sims_q2t = np.zeros((vl_feats.size(0), y))  # 预分配最终结果矩阵
            for i in range(0, vl_feats.size(0), batch_size_q):
                # 分批加载查询特征（CPU → GPU）
                vl_batch = vl_feats[i:i+batch_size_q].to(device)  # (100, 256)
                
                for j in range(0, y, batch_size_t):
                    # 分批加载交叉特征（直接从磁盘映射）
                    cross_batch = torch.from_numpy(
                        cross_feats[i:i+batch_size_q, j:j+batch_size_t]
                    ).permute(1,0,2).to(device)  # (80, 32, 256)
                    
                    # 计算局部相似度
                    sim_batch = sim_matrix_training(vl_batch, cross_batch, 'max')  # (100, 100)
                    
                    # 填充到最终矩阵
                    sims_q2t[i:i+batch_size_q, j:j+batch_size_t] = sim_batch.cpu().numpy()
                    
                    del cross_batch, sim_batch
                    torch.cuda.empty_cache()
                    gc.collect()
            # sims_q2t = torch.cat(sims_q2t, dim=0) 
            print(f'sims_q2t的形状：{sims_q2t.shape}')
            #========清理磁盘文件============================
            os.remove('/media/bd/PSSD/cross_feats.mmap') 
            del vl_feats
            gc.collect()
            #=======================================================

            # sims_q2t = torch.einsum("iqe,jke->ijqk", vl_feats, tar_feats)
            # Process in batches to avoid memory issues
            # batch_size = 100
            # sims_q2t = []
            # for i in range(0, vl_feats.size(0), batch_size):
            #     vl_feats_batch = vl_feats[i : i + batch_size]
            #     print(f'vl_feats_batch的形状：{vl_feats_batch.shape}')
            #     sim_batch = torch.einsum("iqe,jke->ijqk", vl_feats_batch, tar_feats)
            #     print(f'sim_batch的形状：{sim_batch.shape}')
            #     sims_q2t.append(sim_batch)
            # sims_q2t = torch.cat(sims_q2t, dim=0)
            # print(f'sims_q2t的形状：{sims_q2t.shape}')
            # sims_q2t = sims_q2t.max(dim=-1)[0]
            # sims_q2t = sims_q2t.max(dim=-1)[0]
            
            # Create a mapping from pair_id to row index for faster lookup
            pairid2index = {pair_id: i for i, pair_id in enumerate(pair_ids)}

            # Create a mapping from target_id to column index for faster lookup
            tarid2index = {tar_id: j for j, tar_id in enumerate(id2emb.keys())}

            # Update the similarity matrix based on the condition
            for pair_id in pair_ids:
                que_id = data_loader.dataset.pairid2ref[pair_id]
                if que_id in tarid2index:
                    sims_q2t[pairid2index[pair_id], tarid2index[que_id]] = -100
            # sims_q2t = sims_q2t.cpu().numpy()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = {}
            recalls["version"] = "rc2"
            recalls["metric"] = "recall"

            recalls_subset = {}
            recalls_subset["version"] = "rc2"
            recalls_subset["metric"] = "recall_subset"

            target_imgs = np.array(list(id2emb.keys()))

            assert len(sims_q2t) == len(pair_ids)
            for pair_id, query_sims in zip(pair_ids, sims_q2t):
                sorted_indices = np.argsort(query_sims)[::-1]

                query_id_recalls = list(target_imgs[sorted_indices][:50])
                recalls[str(pair_id)] = query_id_recalls

                members = data_loader.dataset.pairid2members[pair_id]
                query_id_recalls_subset = [
                    target
                    for target in target_imgs[sorted_indices]
                    if target in members
                ][:3]
                recalls_subset[str(pair_id)] = query_id_recalls_subset

            json_dump(recalls, "recalls_cirr.json")
            json_dump(recalls_subset, "recalls_cirr_subset.json")

            print(f"Recalls saved in {Path.cwd()}/recalls_cirr.json")

        fabric.barrier()
