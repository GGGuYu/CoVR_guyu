"""
柱状图
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

import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# 数据准备
new_results = {"R1": 62.44, "R5": 86.32, "R10": 92.56}
original_results = {"R1": 61.84, "R5": 86.24, "R10": 92.40}
metrics = ['R1', 'R5', 'R10']
x = np.arange(len(metrics))
width = 0.35  # 柱宽

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# 绘制渐变柱状图
gradient = np.linspace(0, 1, 256).reshape(1, -1)
bars1 = ax.bar(x - width/2, 
              [new_results[m] for m in metrics],
              width,
              label='新损失函数',
              color=plt.cm.Blues(gradient[:, 150]),
              edgecolor='black',
              linewidth=0.8,
              yerr=[0.02]*3,  # 添加误差线
              capsize=4)

bars2 = ax.bar(x + width/2, 
              [original_results[m] for m in metrics],
              width,
              label='原损失函数',
              color=plt.cm.Oranges(gradient[:, 150]),
              edgecolor='black', 
              linewidth=0.8,
              yerr=[0.02]*3,
              capsize=4)

# 添加数值标签
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10,
                    color='black')
autolabel(bars1)
autolabel(bars2)

# 精细化坐标轴
ax.set_ylabel('准确率 (%)')
ax.set_title('损失函数对比 (Δ值放大显示)', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(60, 92.5)  # 聚焦差异区域
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper center', ncol=2, framealpha=0.9)

# 保存图片
plt.savefig('loss_comparison.png', bbox_inches='tight', pad_inches=0.1)
plt.close()
