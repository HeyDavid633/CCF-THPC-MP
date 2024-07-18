import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

# 加载数据
df = pd.read_csv('ablation_data_accuracy.csv')

# 设置全局字体为自定义字体
font_path = '../Times2.ttf'
prop1 = fm.FontProperties(fname=font_path, size=24)
prop2 = fm.FontProperties(fname=font_path, size=20)

# 准备数据
models = df['Model'].values.tolist()
stage1_only = df['only Stage1'].str.rstrip('%').astype(float).values.tolist()  # 去掉百分号并转换为浮点数
stage2_only = df['only Stage2'].str.rstrip('%').astype(float).fillna(0).values.tolist()  # 同上，处理NaN为0
stage1_2 = df['Stage1 + Stage2'].str.rstrip('%').astype(float).fillna(0).values.tolist()  # 同上，处理NaN为0

# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bar_width = 0.25
index = np.arange(len(models))
bar1 = ax1.bar(index, stage1_only, bar_width*0.85, color='#FFA543', label='Only Stage 1')
bar2 = ax1.bar(index + bar_width, stage2_only, bar_width*0.85, color='#FF7F0E', label='Only Stage 2')
bar3 = ax1.bar(index + 2 * bar_width, stage1_2, bar_width*0.85, color='#729DC7', label='Stage 1 + Stage 2')

# 设置Y轴刻度和标签
ax1.set_ylabel('Top-1 Accuracy (%)', fontproperties=prop1)
ax1.set_ylim(0, 100)  # 设置Y轴上限为100
ax1.set_yticks(np.arange(0, 101, 20))  # 设置Y轴刻度从0开始，间隔为20
ax1.set_yticklabels(['{:.0f}'.format(y) for y in np.arange(0, 101, 20)])  # 显示整数
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(models, fontproperties=prop2)

plt.yticks(fontproperties=prop1)

# 添加图例
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), prop=prop2, ncols=3)

# 设置标题和横轴标签
ax1.set_xlabel('Model', fontproperties=prop1)

plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("evaE_ablation_accuracy.pdf")