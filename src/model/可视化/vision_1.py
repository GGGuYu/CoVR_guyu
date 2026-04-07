"""
折线图
多模态对齐损失和普通硬辅对比损失的消融
相关实验参数(没有使用两个前置dyt)：
        if epoch <= 2:  
            pass
        elif 2 < epoch <= 4:
            # self.loss.alpha = 1.0
            # self.loss.beta = 0.5
            # self.loss.tau_vis = 0.6
            # self.loss.tau_txt = 0.45
            # self.loss.gamma = 2.0
            pass
        elif epoch > 4:
            # self.loss.alpha = 1.0
            # self.loss.beta = 0.5
            # self.loss.tau_vis = 0.8
            # self.loss.tau_txt = 0.65
            # self.loss.gamma = 2.5
            pass
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_data(data_str):
    return eval(data_str.replace("'", "\""))

# 数据准备（同上）
custom = [
    parse_data('''{"R1": 61.12, "R5": 85.44, "R10": 91.6}'''),
    parse_data('''{"R1": 61.84, "R5": 85.92, "R10": 92.0}'''),
    parse_data('''{"R1": 62.2, "R5": 85.8, "R10": 92.6}'''),
    parse_data('''{"R1": 61.68, "R5": 86.0, "R10": 92.52}'''),
    parse_data('''{"R1": 62.32, "R5": 85.96, "R10": 92.44}'''),
    parse_data('''{"R1": 62.12, "R5": 86.08, "R10": 92.4}''')
]

normal = [
    parse_data('''{"R1": 61.12, "R5": 85.44, "R10": 91.6}'''),
    parse_data('''{"R1": 61.84, "R5": 85.92, "R10": 92.0}'''),
    parse_data('''{"R1": 62.16, "R5": 85.8, "R10": 92.64}'''),
    parse_data('''{"R1": 61.68, "R5": 86.0, "R10": 92.6}'''),
    parse_data('''{"R1": 62.28, "R5": 86.0, "R10": 92.44}'''),
    parse_data('''{"R1": 62.2, "R5": 86.08, "R10": 92.4}''')
]

# 新增每个指标的纵坐标范围配置
Y_LIM_CONFIG = {
    'R1': (60.8, 63.0),
    'R5': (85.2, 87.0),
    'R10': (90.8, 93.5)
}

# 创建可视化
plt.figure(figsize=(14, 9), dpi=120)
metrics = ['R1', 'R5', 'R10']

for i, metric in enumerate(metrics, 1):
    ax = plt.subplot(3, 1, i)
    x = range(1,7)
    
    c_data = [d[metric] for d in custom]
    n_data = [d[metric] for d in normal]
    
    # 绘制曲线（同上）
    ax.plot(x, c_data, marker='o', markersize=10, color='#FF6B6B', linewidth=3, alpha=0.9)
    ax.plot(x, n_data, marker='^', markersize=10, color='#4ECDC4', linewidth=3, alpha=0.9)
    
    # 设置精细化纵坐标
    y_min, y_max = Y_LIM_CONFIG[metric]
    ax.set_ylim(y_min, y_max)
    
    # 生成更密集的网格线（步长0.5%）
    y_ticks = np.arange(y_min, y_max + 0.1, 0.5)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    
    # 差异标注（调整到新坐标系）
    diffs = np.abs(np.array(c_data) - np.array(n_data))
    if np.max(diffs) > 0.1:
        max_idx = np.argmax(diffs)
        ax.annotate(f'Δ={diffs[max_idx]:.2f}', 
                    xy=(x[max_idx], max(c_data[max_idx], n_data[max_idx])),
                    xytext=(0, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='grey', lw=1))

plt.tight_layout()
plt.savefig('./loss_comparison_refined.png', bbox_inches='tight')
print("=> 优化版可视化已保存至: loss_comparison_refined.png")

