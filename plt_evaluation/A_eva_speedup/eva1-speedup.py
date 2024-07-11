# 7.10 对于eva1 speedup-bar的作图 更新
# 
# 字体也需要是我所要求的字体
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置全局字体为自定义字体，增加字号
font_path = '../Times2.ttf'
fontprop1 = fm.FontProperties(fname=font_path, size=18)
fontprop2 = fm.FontProperties(fname=font_path, size=16)

df = pd.read_csv('./A100_eva1_speedup.csv')

# 计算加速比
df['FP32_ratio'] = df['FP32'] / df['FP32']
df['AMP_ratio'] = df['FP32'] / df['AMP']
df['EMP_ratio'] = df['FP32'] / df['EMP']

# 颜色定义
colors = ['#9DC3E6', '#A9D18E', '#FFA543']

# 创建图表
fig, ax = plt.subplots(figsize=(20, 4))


# 定义X轴位置
width = 0.25
x_pos = range(len(df))

# for idx, column in enumerate(['FP32', 'AMP', 'EMP']):
#     bars = ax.bar([p + width * idx for p in x_pos], df[f'{column}_ratio'], width*0.8, label=column, color=colors[idx], edgecolor='black')
    
#     # 在柱状图顶部添加数值
#     for i, bar in enumerate(bars):
#         yval = df.iloc[i][column]  # 直接从原始列读取
#         if yval < 0.1:
#             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{yval*100:.2f}", ha='center', va='bottom', fontproperties=fontprop2)
#         else:
#             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{yval:.2f}", ha='center', va='bottom', fontproperties=fontprop2)


# 绘制柱状图
for idx, column in enumerate(['FP32_ratio', 'AMP_ratio', 'EMP_ratio']) :
    bars = ax.bar([p + width * idx for p in x_pos], df[column], width*0.8, label=column.replace('_ratio', ''), color=colors[idx], edgecolor='black')
    
    # 在柱状图顶部添加数值
    for i, bar in enumerate(bars):
        yval = df.iloc[i][column]
        ax.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}x", ha='center', va='bottom', fontproperties=fontprop2)

# 设置X轴
ax.set_xticks([p + width * 1 for p in x_pos])
ax.set_xticklabels(df['Net'], ha='center', fontproperties=fontprop1)

# 定义Y轴刻度
yticks = [0.0, 0.5, 1.0, 1.5, 2.0]
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontproperties=fontprop2)
ax.set_ylabel('Training Speedup', fontproperties=fontprop1)

# 添加网格线
ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

# 添加图例
ax.legend(prop=fontprop1, framealpha=0.6, ncol = 3)

# 在每组柱状图之间添加细线
for i in range(0, len(df) - 1):
    ax.axvline(x=x_pos[i] + 3*width, color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)


# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig("eva1_A100_Speedup_ratio.pdf")