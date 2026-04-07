"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from src.model.blip2.blip2 import Blip2Base, disabled_train
from src.tools.utils import all_gather_with_grad, concat_all_gather
from src.model.cloud.cloud import Cloud
# from src.model.blip2.video_transformer import Transformer
from src.model.blip2.DyT import DynamicTanh , convert_ln_to_dyt
# from src.model.blip2.f2seq_tf import Transformer as F2SeqTF
from src.model.blip2.xpool_cross_att import Transformer as xpool_cross_att
from src.model.blip2.xpool_cross_att import sim_matrix_training
# from src.model.blip2.SubspaceTransformer import SubspaceTransformer
from src.model.blip2.ca_for_mask import CrossTransformer

class BLIP2Cir(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        loss: Any,
        vit_model="clip_L",
        image_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        train_vit=False,
        vit="large",
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        temperature=1,
        si_ti_weight=1,
        si_tc_weight=0,
    ):
        super().__init__()

        self.loss = loss

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.train_vit = train_vit
        if not train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = temperature

        self.max_txt_len = max_txt_len

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        for p in self.ln_vision.parameters():
            p.requires_grad = False

        for p in self.Qformer.cls.parameters():
            p.requires_grad = False

        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight

        # # 实例化两个网络
        # diff_calculator = F2SeqTF(
        #     embed_dim=768,
        #     num_heads=12,
        #     num_layers=12,
        #     dropout=0.1
        # ) # feat + seq => seq
        # self.diff_calculator = convert_ln_to_dyt(diff_calculator)
        # final_fuser = F2SeqTF(
        #     embed_dim=embed_dim,
        #     num_heads=8,
        #     num_layers=8,
        #     dropout=0.1
        # ) # feat + seq => seq
        # self.final_fuser = convert_ln_to_dyt(final_fuser)
        # self.diff_loss_weight=0.2 # 添加新的权重参数，可以调整默认值,损失配比

        #改动 9 使用传统交叉注意力,我去适配它
        # self.diff_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        #改动 11 xpool交叉注意力
        self.xpool_cross_att = xpool_cross_att(embed_dim=embed_dim, num_heads=1)
        # ca = CrossTransformer(embed_dim , 1)
        # self.ca = convert_ln_to_dyt(ca)
        # ---xpool_cross_att的替代模块
        # subspace_dim = 8
        # self.subspaceTransformer = SubspaceTransformer(embed_dim=embed_dim, num_heads=8 ,subspace_dim=subspace_dim)

        #给原视频信息的投影
        # self.ref_img_proj = nn.Linear(1408, 1024)
        #cat后投影
        # self.cat_proj_0 = nn.Linear(8*embed_dim, 4*embed_dim)
        # self.cat_proj_1 = nn.Linear(4*embed_dim, 2*embed_dim)
        # self.cat_proj_2 = nn.Linear(2*embed_dim, embed_dim)
        # self.cat_proj_edit = nn.Linear(768, embed_dim)
        # self.cat_proj_img = nn.Linear(1408, embed_dim)
        #--------------------------------------------------------------------------

    def forward(self, batch, fabric):
        #改动
        #-----------------------------------------------
        # txt1 = batch["txt1"]
        #-----------------------------------------------
        ref_img = batch["ref_img"]
        # print(f'ref_img的形状: {ref_img.shape}') #64 ,3 ,3 ,364 ,364
        # ref_img = ref_img.reshape(-1  ,3 ,364 ,364)

        tar_img_feat = batch["tar_img_feat"]
        caption = batch["edit"]

        ref_img.half()

        device = ref_img.device

        # Encode the target image
        tar_img_feat = tar_img_feat.to(device)
        tar_img_feat = concat_all_gather(tar_img_feat, fabric)

        # Text
        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        #改动8 将txt全部token化
        #----------------------------------------------------------------------------------------------------------------------------------------
        # 原始视频文本描述 txt1 
        # orgin_vid_text_tokens = self.tokenizer(
        #     txt1,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(device)
        #----------------------------------------------------------------------------------------------------------------------------------------
        if self.train_vit:
            ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
        else:
            with torch.no_grad():
                ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))

        # Encode the reference image
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        ###============== Image-text Matching ===================###
        query_tokens = self.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        output = self.Qformer.bert(
            text_tokens.input_ids,  # [bs, 32]
            query_embeds=query_tokens,  # [bs, 32, 768]
            attention_mask=attention_mask,  # [bs, 64]
            encoder_hidden_states=ref_img_embs,  # [bs, 677, 1408]
            encoder_attention_mask=ref_img_atts,  # [bs, 677]
            return_dict=True,
        )

        vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :] #[bs , 32 , 768]
        query_si_feat = F.normalize(self.text_proj(vl_embs), dim=-1) #[bs ,32 ,256]
        query_si_feat = all_gather_with_grad(query_si_feat, fabric)

        # mean over all query tokens 改动5暂时注释掉平均
        # query_si_feats = query_si_feat #我保留一下mean之前的结果
        query_si_feat = query_si_feat.mean(dim=1) #[bs , 256]
        query_si_feat = F.normalize(query_si_feat, dim=-1) #真正的查询
        # tar_img_feat = tar_img_feat.mean(dim=1)
        tar_img_feat = F.normalize(tar_img_feat, dim=-1)
        # print(f'不mean的tarf:{tar_img_feat.shape}')

        # #改动8,学习修改文本和原始描述文本之间的差异
        # #-------------------------------------------------------------------------
        # # --- 步骤 1: 单独编码文本特征 (类似 blip2_cir_text.py) ---
        # 编码修改指令文本
        # edit_query_embs = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # ) # [bs, 32, 768]
        
        # edit_query_embs = edit_query_embs.last_hidden_state[:, :, :]
        # ca_img = self.cat_proj_img(ref_img_embs) #(B,677,256)
        # ca_img = F.normalize(ca_img, dim=-1)
        # ca_edit = self.cat_proj_edit(edit_query_embs) #(B,32,256)
        # ca_edit = F.normalize(ca_edit, dim=-1)

        # guide_q = self.ca(ca_img , ca_edit)
        # guide_q = F.normalize(guide_q, dim=-1)
   
        # add_frame_edit = edit_query_embs.last_hidden_state[:, 0, :] # 取 [CLS] token #全局特征 [bs, 768]
        # add_frame_edit = self.cat_proj_1(add_frame_edit) #768变256
        # add_frame_edit = F.normalize(add_frame_edit, dim=-1).unsqueeze(1) #(B,1,D)
        # print(f'add_frame_edit的形状:{add_frame_edit.shape}')
        
        

        # ==========原视频信息 todo=============
        # add_frame_img = ref_img_embs[:, 0, :] #(bs , 1408)
        # add_frame_img = self.cat_proj_2(add_frame_img)#(bs,256)
        # add_frame_img = F.normalize(add_frame_img, dim=-1).unsqueeze(1) #(B,1,D)
        # print(f'add_frame_img的形状:{add_frame_img.shape}')
        # =================================

        
        #-------------------------------------------------------------------------
        
        #cat
        # print(query_si_feat.shape)
        # print(edit_vl_embs.shape)
        # combined = torch.cat([query_si_feat, edit_vl_embs], dim=1) #查询和修改文本cat 1024
        # combined = torch.cat([combined, ref_img_embs_cls], dim=1) #1024+1024=2048
        # combined = self.cat_proj_0(combined) #1024
        # print(combined.shape)

        # combined = self.cat_proj_1(combined) #512
        # query_si_feat_cross = self.cat_proj_2(combined)#256 只是用作特化数据库的指导信息，并不是真正的查询
        # query_si_feat_cross = F.normalize(query_si_feat_cross, dim=-1)
        #计算xpool_hn_nce需要的对比矩阵
        #---------------------------------
        # tar_img_feat_add_frame = tar_img_feat.unsqueeze(1) #变成(B , 1 ,dim)
        # tar_combined = torch.cat([add_frame_img,tar_img_feat_add_frame,add_frame_edit], dim=1)
        # print(f'三帧的形状:{tar_combined.shape}')
        # cross_feats,attn_entropy = self.xpool_cross_att(query_si_feat , tar_img_feat_add_frame,is_training=True) #交叉结果(B,B,dim)
        # cross_feats = self.subspaceTransformer(query_si_feat_cross , tar_img_feat_add_frame) #交叉结果(B,B,dim)
        # xpool_sim_matrix = sim_matrix_training(query_si_feat , cross_feats,'max') #(B,B)对比矩阵
        # cross_feats , attn_entropy = self.xpool_cross_att(query_si_feat , tar_img_feat,is_training=True) #交叉结果(B,B,dim)
        # --------
        # cross_feats, attn_entropy = self.xpool_cross_att(guide_q , tar_img_feat,is_training=True) #交叉结果(B,B,dim)
        cross_feats = self.xpool_cross_att(query_si_feat , tar_img_feat,is_training=False) #交叉结果(B,B,dim)
        xpool_sim_matrix = sim_matrix_training(query_si_feat , cross_feats,'max') #(B,B)对比矩阵
        #---------------------------------



        #loss计算
        # s=source, t=target, i=image, c=caption, w=weight
        loss = 0
        if self.si_ti_weight > 0:
            #改动，使用xpool_hn_hce损失
            # print("测试损失")
            si_ti_loss = self.loss(xpool_sim_matrix,self.temp)
            loss += si_ti_loss * self.si_ti_weight

            # ===== 加上那个注意力熵 =====
            # lambda_ent = 0.2
            # loss = loss + lambda_ent * attn_entropy
            # ============================


        use_txt_loss = False
        if self.si_tc_weight > 0:
            # print(f'检测是否打开了txt_loss')
            assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
            tar_txt_feat = batch["tar_txt_feat"]

            tar_txt_feat = all_gather_with_grad(tar_txt_feat, fabric)
            #改动
            #------------------------------------------------------
            #云模型
            # print(f'文本加噪，因为文本信息太薄弱，因此加入基于云模型的不确定性，未使用可学习参数')
            # print(f'self.si_tc_weight={self.si_tc_weight}')
            cloud = Cloud(tar_txt_feat, 1 ,tar_txt_feat.shape[1])
            tar_txt_feat = cloud.get_cloud()
            #------------------------------------------------------

            si_tc_loss = self.loss(query_si_feat, tar_txt_feat, self.temp)
            # loss += si_tc_loss * self.si_tc_weight
            # 改动
            # ------------------------------------------------------
            loss += si_tc_loss * si_tc_w
            # ---------------------------------
        
        #改动
        #-------------------------------------------------------------
        #增加diff_feat的loss,希望他和tar_txt_feat尽量像，或者和tar_img_feat这个尽量像，其实也可以
        is_diff_loss = False
        if is_diff_loss:
            diff_feat_loss = self.loss(diff_feat, tar_img_feat, self.temp)
            loss = (diff_feat_loss * self.diff_loss_weight) + loss

        return loss


def blip2_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model


# ------------------------------------------------------------------------------------------------
# 调试信息对齐
# txt1的值 = ['a man playing guitar close up', 'austria montafon summer', 'female model posing in underwear in studio slow motion', 'cooked bacon strips on a baking sheet', 'frying mackerel fish in pan', 'handsome young man laughs', 'mature businessman with tablet in the office', 'rocks in sea', 'laboratory soil analysis', 'paper texture', 'crystal clear water', 'letters are collected in text hello, then scattered into strips bright colors alpha channel premultiplied - matted with color black', 'kidney in analysis', 'close up fresh raspberry', 'mountain river in georgia', 'hearts draw patterns on the screen', 'cow stables close up', 'intercom call', 'air bubbles in the liquid', 'taipei city, taiwan, 27 may 2018 - taipei city wall', 'zooming to tanzania on political globe 3d illustration', 'aerial view of a russian landscape', 'lions in the savannah', 'woman use of mobile phone at outdoor', 'ruins, old house', 'collection of different shots about nature, animals and rest in the mountains of the carpathian, ukraine', 'hiking in autumn forest', 'takeover definition', 'beef cooking on a fireplace', 'thoracic spine with ligaments, blood vessels and nerves', 'abstract water texture close up', 'white lamb closeup', 'medicine definition', 'green field', "lyle's flying foxes pteropus lylei hangs on a tree branch and washes", 'lake in the forest nature', 'the wild bird', 'symbol cookie burns out of transparency, then burns again alpha channel premultiplied - matted with color black', 'determined senior caucasian woman doing pull-ups in fitness studio 4k', 'young man texting on smartphone and drinking cocktail in cafe', 'portrait of a man', 'amusement park ride', 'saint-petersburg, russia - may, 2015 gorgeous rooms and interiors of the catherine palace in st petersburg pushkin tsarskoye selo', 'girl laughing', 'dance of colors', 'flying over the snowy forest', 'brunette girl puts on make up', 'rice milling machine', 'portrait of smiling little girl', 'close-up view of green splashes in slow motion juice', 'the man driving a car on the night city', 'golden money', 'dance of color grids on the screen', 'flash light space', 'rolling waves in the ocean', 'rough sea', 'woman playing with baby', 'reindeer village on an summer afternoon in tsaatan village, khuvsgol, mongolia', 'veggies in a glass bowl', 'musician playing on violin', 'simple white birthday cake with cake garland', 'delos more ruins', 'young girl riding a horse in a forest', 'couple walking on the beach']
# txt2的值 = ['asian man playing guitar close up', 'austria montafon hut', 'female model posing in shirt in studio slow motion', 'cooked bacon strips on a bacon sheet', 'frying tilapia fish in pan', 'handsome young man smiles', 'young businessman with tablet in the office', 'breakwater in sea', 'laboratory test analysis', 'soap texture', 'crystal clear stream', 'letters are collected in text pure, then scattered into strips bright colors alpha channel premultiplied - matted with color black', 'map in analysis', 'close up fresh strawberry', 'mountain stream in georgia', 'tapes draw patterns on the screen', 'cow herd close up', 'business call', 'air bubbles in the water', 'taipei city, taiwan, 27 may 2018 - taipei city street', 'zooming to namibia on political globe 3d illustration', 'aerial view of a green landscape', 'lions in the serengeti', 'woman use of mobile phone at home', 'movie old house', 'collection of different shots about nature, animals and training in the mountains of the carpathian, ukraine', 'mushrooms in autumn forest', 'terrorism definition', 'beef cooking on a griddle', 'cervical spine with ligaments, blood vessels and nerves', 'abstract glass texture close up', 'white flower closeup', 'workflow definition', 'green coaster', "lyle's flying fox pteropus lylei hangs on a tree branch and washes", 'stream in the forest nature', 'nightingale wild bird', 'symbol store burns out of transparency, then burns again alpha channel premultiplied - matted with color black', 'determined senior caucasian woman doing plank in fitness studio 4k', 'young man texting on smartphone and drinking beer in cafe', 'portrait of smiling man', 'amusement park attraction', 'saint-petersburg, russia - may, 2015 gorgeous ceilings and interiors of the catherine palace in st petersburg pushkin tsarskoye selo', 'girl dances', 'patterns of colors', 'flying over the winter forest', 'beautiful girl puts on make up', 'dental milling machine', 'portrait cute smiling little girl', 'close-up view of purple splashes in slow motion juice', 'the man driving a car on the night road', 'golden coins', 'dance of color patterns on the screen', 'spot light space', 'storm waves in the ocean', 'the sea', 'mother playing with baby', 'reindeer village on an summer morning in tsaatan village, khuvsgol, mongolia', 'eggs in a glass bowl', 'woman playing on violin', 'simple white birthday cake with cake candles', 'delos house ruins', 'young girl riding a horse in a meadow', 'seagulls walking on the beach']
# edit的值 = ['making the man asian', 'make it a hut', 'make her wear a shirt', 'make it into bacon', 'replace the mackerel with tilapia', 'have him smile', 'have a young business', 'make it a breakwater', 'put the soil', 'make it look like soap', 'make the water a stream', 'make the letters pure, and make the letters black', 'turn it into a map', 'make it a straw', 'make it a stream', 'have the tapes draw', 'make it a herd', 'change to a business call', 'make the liquid into water', 'turn the background into a street', 'move to namibia', 'make it a green landscape', 'in the savannah', 'put it on the table', 'replace with a movie old house', 'the new shots should be about training in the mountains of the carpathian, ukraine', 'have the mushrooms', 'make it a terror', 'make it a griddle', 'make the thoracic spine the cervical spine', 'make it into glass', 'replace the lamb with a flower', 'make the definition', 'have it as a green coaster', 'make it a flying fox', 'make it look like a stream', 'turn it into a nightingale', 'matting it with color black', 'have her do the plank', 'replace the cocktail with a beer', 'make the man smiling', 'to be an attraction', 'tren', 'make her dance', 'as a pattern', 'make it winter', 'make her more', 'make it a dental milling machine', 'make her more adorable', 'make the splashes purple', 'make it a night road', 'make the money into coins', 'add color patterns', 'make the light a spot light', 'change from rolling to storm waves', 'make the sea', 'have the mother play with', 'make it a morning', 'make the eggs', 'make it a woman', 'make it a birthday cake candle instead of a garland', 'make it a', 'have her riding a horse in a meadow', 'make the couple seagulls']