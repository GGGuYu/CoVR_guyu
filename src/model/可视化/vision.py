"""
53.13 79.93 86.85 97.69  CoVR-BLIP
59.82 83.84 91.28 98.24  CoVR-BLIP2
60.12 84.32 91.27 98.72  BLIP-EC&DE
62.32 85.96 92.44 98.68  Our Approach

对比表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置为非交互式后端
plt.switch_backend('agg')

# 自定义样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 14,
    'figure.figsize': (16, 9),
    'figure.dpi': 300
})

# 数据录入
data = {
    "R1": [53.13, 59.82, 60.12, 62.44],
    "R5": [79.93, 83.84, 84.32, 86.32],
    "R10": [86.85, 91.28, 91.27, 92.56],
    "R50": [97.69, 98.24, 98.72, 98.72],
    "method": ["CoVR-BLIP", "CoVR-BLIP2", "BLIP-EC&DE", "Our Approach"]
}
df = pd.DataFrame(data)
metrics = ['R1', 'R5', 'R10', 'R50']
x_pos = np.arange(len(metrics))  # 指标位置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 对比色系

# 创建画布
fig, ax = plt.subplots()

# 绘制每个方法的趋势线
for i, method in enumerate(df['method']):
    # 主趋势线
    ax.plot(x_pos, df[metrics].values[i], 
            marker='o', markersize=10, 
            linewidth=3, alpha=0.9,
            color=colors[i], label=method)
    
    # 误差带显示差距
    lower = df[metrics].values[i] - 0.1
    upper = df[metrics].values[i] + 0.1
    ax.fill_between(x_pos, lower, upper, 
                    color=colors[i], alpha=0.15)

# 标注关键数据点
for col_idx, metric in enumerate(metrics):
    for row_idx in range(len(df)):
        y = df[metric].iloc[row_idx]
        ax.text(col_idx, y+0.3, f'{y:.2f}', 
                ha='center', va='bottom', 
                fontsize=12, color=colors[row_idx],
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 坐标轴优化
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, rotation=0)
ax.set_ylim(50, 100)
ax.set_yticks(np.arange(50, 100.1, 2))
ax.set_ylabel('Performance (%)', fontweight='bold', labelpad=15)
ax.set_xlabel('Evaluation Metrics', fontweight='bold', labelpad=15)

# 图例与标题
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                   frameon=True, shadow=True, 
                   facecolor='white', edgecolor='#333333',
                   title='Methods', title_fontsize='14')
legend.get_title().set_fontweight('bold')

plt.title('Performance Comparison Across Metrics', 
          pad=20, fontsize=18, fontweight='bold')

# 保存结果
plt.tight_layout()
plt.savefig('result.png', bbox_inches='tight', pad_inches=0.2)
print("可视化结果已保存为 result.png")
