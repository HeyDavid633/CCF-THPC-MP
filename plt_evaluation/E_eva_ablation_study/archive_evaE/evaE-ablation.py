import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

# 加载数据
df = pd.read_csv('CoMP_ablation_study.csv')

# 设置全局字体为自定义字体
font_path = '../Times2.ttf'
prop1 = fm.FontProperties(fname=font_path, size=24)
prop2 = fm.FontProperties(fname=font_path, size=20)

# 准备数据
models = df['Model'].values.tolist()
basis = df['basis'].values.tolist()
normalized_time = df['normalized-time'].values.tolist()
speedup = df['speedup'].values.tolist()

# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bar_width = 0.35
index = range(len(models))
bar1 = ax1.bar(index, basis, bar_width * 0.85, color='#FFA543', label='Only Stage 1')
bar2 = ax1.bar([i + bar_width for i in index], normalized_time, bar_width * 0.85, color='#FF7F0E', label='Stage 1 + Stage 2')

# 设置Y轴刻度和标签
ax1.set_ylabel('Normalized Time', fontproperties=prop1)
ax1.set_ylim(0.8, 1.1)
ax1.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10])
ax1.set_xticks([i + bar_width / 2 for i in index])
ax1.set_xticklabels(models, fontproperties=prop2)

plt.yticks(fontproperties=prop1)

# 绘制折线图
# def custom_formatter(y, pos):
#     return f"{y:.1f}x"

# ax2 = ax1.twinx()
# line_x = [i + bar_width / 2 for i in index]
# line = ax2.plot(line_x, speedup, 'o-', color= '#729DC7', markersize=8, label='Speedup')
# ax2.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
# ax2.set_ylabel('Speedup', fontproperties=prop1)
# ax2.set_ylim(0.8, 1.2)
# ax2.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
# plt.yticks(fontproperties=prop1)

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.19), prop=prop2, ncols=3)
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.19), prop=prop2, ncols=3)

# 设置标题和横轴标签
# ax1.set_title('CoMP Ablation Study', fontproperties=fontprop)
ax1.set_xlabel('Model', fontproperties=prop1)

plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("evaE_ablation_study.pdf")