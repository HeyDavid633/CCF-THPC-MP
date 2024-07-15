import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties

font_path = '../Times2.ttf'
prop1 = FontProperties(fname=font_path, size=20)
prop2 = FontProperties(fname=font_path, size=24)

width = 0.8
color1 = '#91CBFC'  # 有点亮的浅蓝
color3 = '#729DC7'  # 深一点的浅蓝

# 读取CSV文件
df = pd.read_csv("CoMP_overhead.csv")
df['CoMP Stage1'] = df['CoMP Stage1'].str.rstrip('%').astype(float)
df['CoMP Stage2'] = df['CoMP Stage2'].str.rstrip('%').astype(float)
df['Train'] = 100 - (df['CoMP Stage1'] + df['CoMP Stage2'])  # 动态计算Train列的值
models = df['Model'].tolist()

plt.figure(figsize=(10, 6))

# 绘制堆叠条形图，顺序调整为CoMP Stage1 -> CoMP Stage2 -> Train
plt.bar(models, df['CoMP Stage1'], bottom=df['Train'] + df['CoMP Stage2'], color='#FFA543',  width=width*0.82, label='CoMP Stage 1')
plt.bar(models, df['CoMP Stage2'], bottom=df['Train'], color='#FF7F0E', width=width*0.82, label='CoMP Stage 2')
plt.bar(models, df['Train'], color=color3, width=width*0.82, label='Post-Trainig')

# # 调整数值标签的位置
# for i, model in enumerate(models):
#     # CoMP Stage1的标签
#     plt.text(i, df.loc[i, 'Train'] + df.loc[i, 'CoMP Stage2'] + df.loc[i, 'CoMP Stage1']*0.2, f'{df.loc[i, "CoMP Stage1"]:.2f}%', ha='center', va='center', color='black', fontproperties=prop1)
    
#     # CoMP Stage2的标签
#     plt.text(i, df.loc[i, 'Train']*0.96, f'{df.loc[i, "CoMP Stage2"]:.2f}%', ha='center', va='center', color='black', fontproperties=prop1)
    
#     # Train的标签
#     plt.text(i, df.loc[i, 'Train']*0.5, f'{df.loc[i, "Train"]:.2f}%', ha='center', va='center', color='black', fontproperties=prop1)

# 设置纵坐标刻度
plt.yticks([0, 25, 50, 75, 100])
plt.ylim(0, 100)

# 设置标题和标签
plt.xlabel('Model', fontproperties=prop2)
plt.ylabel('Percentage (%)', fontproperties=prop2)

# 移动图例位置
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), prop=prop1, ncol = 3)

# 设置坐标轴字体
plt.xticks(fontproperties=prop1)
plt.yticks(fontproperties=prop2)

# 设置网格线
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("evaD_breakdown.pdf")