import datetime
import time
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump
from src.model.blip2.xpool_cross_att import Transformer as xpool_cross_att
from src.model.blip2.xpool_cross_att import sim_matrix_training

class TestWebVidCoVR:
    def __init__(self, remove_self_similarity: bool = True, dataset: str = "covr"):
        self.remove_self_similarity = remove_self_similarity
        self.dataset = dataset

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation...")
        start_time = time.time()

        tar_img_feats = []
        query_feats = []
        captions = []
        pair_ids = []

        #特化数据库的指导信息
        # query_feats_cross = []
        # add_frame_edits = []
        # add_frame_imgs = []

        for batch in data_loader:
            ref_img = batch["ref_img"]
            tar_feat = batch["tar_img_feat"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())
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
            query_feat = F.normalize(model.text_proj(query_feat), dim=-1) # [bs , 32 ,256]

            # 改动5
            #-------------------------------------------------------------------------------------
            # query_si_feats = query_feat #我保留一下mean之前的结果
            # tar_feat = tar_feat.mean(dim=1) #先不mean tar_feat,而是拿去做交叉
            query_feat = query_feat.mean(dim=1)

            #开始你的表演---------------------------------
            query_si_feat = query_feat

            
            # edit_query_embs = model.Qformer.bert(
            #     text_tokens.input_ids,
            #     attention_mask=text_tokens.attention_mask,
            #     return_dict=True,
            # ) # [bs, 32, 768]
    
            # add_frame_edit = edit_query_embs.last_hidden_state[:, 0, :] # 取 [CLS] token #全局特征 [bs, 768]
            # add_frame_edit = model.cat_proj_1(add_frame_edit) #768变256
            # add_frame_edits.append(add_frame_edit.cpu())



            # ==========原视频信息 todo=============
            # add_frame_img = ref_img_embs[:, 0, :] #(bs , 1408)
            # add_frame_img = model.cat_proj_2(add_frame_img)#(bs,256)
            # add_frame_imgs.append(add_frame_img.cpu())
            # =================================

            #cat
            # combined = torch.cat([query_si_feat, edit_vl_embs], dim=1) #1024
            # combined = torch.cat([combined, ref_img_embs_cls], dim=1) #1024+1024=2048
            # combined = model.cat_proj_0(combined) #1024
            # combined = model.cat_proj_1(combined)#只是用作特化数据库的指导信息，并不是真正的查询
            # query_si_feat_cross = model.cat_proj_2(combined)
            # query_feats_cross.append(query_si_feat_cross.cpu())

            
            query_feat = query_si_feat
            #-------------------------------------------------------------------------

            #---------------------------------------------------------------------------------------

            query_feats.append(query_feat.cpu())

            # Encode the target image
            tar_img_feats.append(tar_feat.cpu())
        
        query_feats = torch.cat(query_feats, dim=0) #[2500 ,256] or  [2500 , 32 ,256] #我改了batch中操作以后应该是 [2500 , 256] 原来是 [2500 , 32 ,256]
        tar_img_feats = torch.cat(tar_img_feats, dim=0) #[2500 , 256]   #我想在batch中做好32变1 最后不用在外面做32变1
        # query_feats_cross = torch.cat(query_feats_cross, dim=0)
        # add_frame_imgs = torch.cat(add_frame_imgs , dim=0)
        # add_frame_edits = torch.cat(add_frame_edits , dim=0)

        query_feats = F.normalize(query_feats, dim=-1)
        tar_img_feats = F.normalize(tar_img_feats, dim=-1)
        # query_feats_cross = F.normalize(query_feats_cross, dim=-1)
        # add_frame_imgs = F.normalize(add_frame_imgs, dim=-1).unsqueeze(1) #(B,1,D)
        # add_frame_edits = F.normalize(add_frame_edits, dim=-1).unsqueeze(1) #(B,1,D)

        ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
        tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

        ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
        tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            tar_img_feats = fabric.all_gather(tar_img_feats)
            ref_img_ids = fabric.all_gather(ref_img_ids)
            tar_img_ids = fabric.all_gather(tar_img_ids)

            query_feats = einops.rearrange(query_feats, "d b q e -> (d b) q e")
            tar_img_feats = einops.rearrange(tar_img_feats, "d b q e -> (d b) q e")

            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            # tar_img_feats = tar_img_feats.mean(dim=1) #[2500 , 256]
            # query_feats = query_feats.mean(dim=1) #[2500 , 256]    暂时注释 todo
            print('test测试')
            #---------------------------------
            #计算xpool_hn_nce需要的对比矩阵
            print("计算xpool_hn_nce需要的对比矩阵")
            # tar_img_feat_add_frame = tar_img_feats.unsqueeze(1) #变成(B , 1 ,dim)
            # tar_combined = torch.cat([add_frame_imgs,tar_img_feat_add_frame,add_frame_edits], dim=1)
            original_device = next(model.xpool_cross_att.parameters()).device
            xpool_cross_att = model.xpool_cross_att
            xpool_cross_att.cpu()
            cross_feats = xpool_cross_att(query_feats , tar_img_feats) #交叉结果(B,B,dim)
            model.xpool_cross_att.to(original_device)  # 立即恢复原设备
            # original_device = next(model.subspaceTransformer.parameters()).device
            # subspaceTransformer = model.subspaceTransformer
            # subspaceTransformer.cpu()
            # cross_feats = subspaceTransformer(query_feats_cross , tar_img_feat_add_frame) #交叉结果(B,B,dim)
            # model.subspaceTransformer.to(original_device)  # 立即恢复原设备
            xpool_sim_matrix = sim_matrix_training(query_feats , cross_feats,'max') #(B,B)对比矩阵
            sim_q2t = xpool_sim_matrix.cpu().numpy()
            #---------------------------------
            #sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy() #[2500 , 2500]

            if self.remove_self_similarity:
                for i in range(len(ref_img_ids)):
                    for j in range(len(tar_img_ids)):
                        if ref_img_ids[i] == tar_img_ids[j]:
                            sim_q2t[i][j] = -10

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = eval_recall(sim_q2t)
            recalls["annotation"] = Path(data_loader.dataset.annotation_pth).name
            fabric.print(recalls)

            # Save results
            self_sim = "" if self.remove_self_similarity else "_ss"
            json_dump(recalls, f"recalls_{self.dataset}{self_sim}.json")

            print(
                f"Recalls saved in {Path.cwd()}/recalls_{self.dataset}{self_sim}.json"
            )

        fabric.barrier()
        return recalls


@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean3 = (tr1 + tr5 + tr10) / 3
    tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

    eval_result = {
        "R1": round(tr1, 2),
        "R5": round(tr5, 2),
        "R10": round(tr10, 2),
        "R50": round(tr50, 2),
        "meanR3": round(tr_mean3, 2),
        "meanR4": round(tr_mean4, 2),
    }
    return eval_result
