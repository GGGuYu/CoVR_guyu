            # 假设：
            # vl_feats: (4024, 256)  # 全部文本特征（可保留在内存）
            # cross_feats: (4024, 8048, 256)  # 内存映射的交叉特征
            batch_size_q = 100  # 查询分批大小（根据GPU内存调整）
            batch_size_t = 100  # 目标视频分批大小
            device = gpu
            sims_q2t = np.zeros((vl_feats.size(0), tar_feats.size(0)))  # 预分配最终结果矩阵
            for i in range(0, vl_feats.size(0), batch_size_q):
                # 分批加载查询特征（CPU → GPU）
                vl_batch = vl_feats[i:i+batch_size_q].to(device)  # (100, 256)
                
                for j in range(0, tar_feats.size(0), batch_size_t):
                    # 分批加载交叉特征（直接从磁盘映射）
                    cross_batch = torch.from_numpy(
                        cross_feats[i:i+batch_size_q, j:j+batch_size_t]
                    ).to(device)  # (100, 100, 256)
                    
                    # 计算局部相似度
                    sim_batch = sim_matrix_training(vl_batch, cross_batch, 'max')  # (100, 100)
                    
                    # 填充到最终矩阵
                    sims_q2t[i:i+batch_size_q, j:j+batch_size_t] = sim_batch.cpu().numpy()
                    
                    del cross_batch, sim_batch
                    torch.cuda.empty_cache()
                sims_q2t = torch.cat(sims_q2t, dim=0) 
                print(f'sims_q2t的形状：{sims_q2t.shape}')
                #========清理磁盘文件============================
                os.remove('/media/bd/PSSD/cross_feats.mmap') 
                del vl_feats
                gc.collect()


            batch_size = 100
            sims_q2t = []
            for i in range(0, vl_feats.size(0), batch_size):
                vl_feats_batch = vl_feats[i : i + batch_size]
                print(f'vl_feats_batch的形状：{vl_feats_batch.shape}')
                sim_batch = sim_matrix_training(vl_feats_batch , cross_feats,'max') #(B,B)对比矩阵
                sim_batch = sim_batch.cpu().numpy()
                # 关键步骤：直接写入磁盘（无需保存到内存）
                #sims_q2t[i:i+batch_size] = sim_batch.cpu().numpy()  # 自动分批写入
                print(f'sim_batch的形状：{sim_batch.shape}')
                sims_q2t.append(sim_batch) 
                # 立即释放内存
                del vl_batch, sim_batch
                gc.collect()
            sims_q2t = torch.cat(sims_q2t, dim=0) 
            print(f'sims_q2t的形状：{sims_q2t.shape}')
            del vl_feats,cross_feats
            gc.collect()