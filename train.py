import datetime
import shutil
import time
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.tools.files import json_dump
from src.tools.utils import calculate_model_params


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(Path.cwd())
    
    # 先初始化fabric！！！
    L.seed_everything(cfg.seed, workers=True)
    fabric = instantiate(cfg.trainer.fabric)
    fabric.launch()
    fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    # 新增配置文件检查逻辑（移动到初始化之后）
    if fabric.global_rank == 0:
        config_path = Path(hydra.utils.get_original_cwd()) / "configs/train.yaml"
        fabric.print(f"Loaded config from: {config_path}")
        fabric.print("\nConfiguration content:")
        fabric.print(OmegaConf.to_yaml(cfg))
        
        # 关键配置项检查
        required_fields = ["model", "data", "trainer"]
        for field in required_fields:
            if field not in cfg:
                raise ValueError(f"Missing required config field: {field}")
        
        if "hidden_dim" not in cfg.model:
            fabric.print("⚠️ Warning: hidden_dim not specified in model config")

    if fabric.global_rank == 0:
        json_dump(OmegaConf.to_container(cfg, resolve=True), "hydra.json")

    data = instantiate(cfg.data, _recursive_=False)
    loader_train = fabric.setup_dataloaders(data.train_dataloader())
    if cfg.val:
        loader_val = fabric.setup_dataloaders(data.val_dataloader())

    model = instantiate(cfg.model)
    calculate_model_params(model)

    optimizer = instantiate(
        cfg.model.optimizer, params=model.parameters(), _partial_=False
    )
    model, optimizer = fabric.setup(model, optimizer)

    scheduler = instantiate(cfg.model.scheduler)

    fabric.print("Start training")
    start_time = time.time()

    best_R1 = 0

    for epoch in range(cfg.trainer.max_epochs):
        scheduler(optimizer, epoch)

        columns = shutil.get_terminal_size().columns
        fabric.print("-" * columns)
        fabric.print(f"Epoch {epoch + 1}/{cfg.trainer.max_epochs}".center(columns))

        train(model, loader_train, optimizer, fabric, epoch, cfg)

        # if cfg.val:
            # fabric.print("Evaluate")
            # instantiate(cfg.evaluate, model, loader_val, fabric=fabric)

        state = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        # if cfg.trainer.save_ckpt == "all":
        #     fabric.save(f"ckpt_{epoch}.ckpt", state)
        # elif cfg.trainer.save_ckpt == "last":
        #     fabric.save("ckpt_last.ckpt", state)
        # fabric.barrier()
# 如果是cirr,先不test
        for dataset_t in cfg.test:
            columns = shutil.get_terminal_size().columns
            fabric.print("-" * columns)
            fabric.print(f"Testing on {cfg.test[dataset_t].dataname}".center(columns))

            data_t = instantiate(cfg.test[dataset_t])
            test_loader = fabric.setup_dataloaders(data_t.test_dataloader())

            test_t = instantiate(cfg.test[dataset_t].test)
            recalls = test_t(model, test_loader, fabric=fabric)

            if cfg.trainer.save_ckpt == "best":
                print("检查是否为最佳模型")
                cur_R1 = recalls.get("R1")
                if cur_R1 > best_R1:
                    print("正在保存模型")
                    best_R1 = cur_R1
                    fabric.save("m2_web_16_normal_att_best.ckpt", state)
                    print("保存模型成功")
                else:
                    print("本轮非最佳模型")
            elif cfg.trainer.save_ckpt == "last":
                fabric.save("ckpt_last.ckpt", state)
            fabric.barrier()
#-----
        # print("正在保存模型")
        # # fabric.save("ckpt_fashioniq_last.ckpt", state)
        # fabric.save("ckpt_cirr_last.ckpt", state)
        # fabric.barrier()
        #---

        # if epoch +1 == 5:  
        #     # fabric.save("ckpt_fashioniq_last.ckpt", state)
        #     print("正在保存模型")
        #     name = f"m2_fashioniq_{epoch+1}_n_16_normal.ckpt"
        #     fabric.save(name, state)
        #     fabric.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    fabric.print(f"Training time {total_time_str}")

    # for dataset in cfg.test:
    #     columns = shutil.get_terminal_size().columns
    #     fabric.print("-" * columns)
    #     fabric.print(f"Testing on {cfg.test[dataset].dataname}".center(columns))

    #     data = instantiate(cfg.test[dataset])
    #     test_loader = fabric.setup_dataloaders(data.test_dataloader())

    #     test = instantiate(cfg.test[dataset].test)
    #     test(model, test_loader, fabric=fabric)

    fabric.logger.finalize("success")
    fabric.print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def train(model, train_loader, optimizer, fabric, epoch, cfg):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        if batch == None:
            continue
        optimizer.zero_grad()
        loss = model(batch, fabric)
        fabric.backward(loss)
        optimizer.step()

        if batch_idx % cfg.trainer.print_interval == 0:
            fabric.print(
                f"[{100.0 * batch_idx / len(train_loader):.0f}%]\tLoss: {loss.item():.6f}"
            )
        if batch_idx % cfg.trainer.log_interval == 0:
            fabric.log_dict(
                {
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )


if __name__ == "__main__":
    main()
