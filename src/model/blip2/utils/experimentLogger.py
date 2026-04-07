from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def log_batch(self, epoch, batch_idx, aux_info):
        """记录每个batch的辅助信息"""
        for k, v in aux_info.items():
            self.writer.add_scalar(f'batch_stats/{k}', v, epoch * 1000 + batch_idx)  # 假设每epoch约1000batch
            
    def log_epoch(self, epoch, metrics):
        """记录epoch级指标"""
        for k, v in metrics.items():
            self.writer.add_scalar(f'epoch_stats/{k}', v, epoch)