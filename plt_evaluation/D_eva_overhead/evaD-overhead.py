import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties

# ========== 字体设置 ==========
# 英文字体 (Times New Roman)
times_path = '../Times2.ttf'  # 或系统路径如 '/Library/Fonts/Times New Roman.ttf'
prop_en = FontProperties(fname=times_path, size=20)

# 中文字体 (宋体/SimSun) - Mac 系统路径
simsun_path = '/System/Library/Fonts/Supplemental/Songti.ttc'  # Mac 宋体
# 备选路径:
# simsun_path = '/Library/Fonts/SimSun.ttc'
# simsun_path = '/System/Library/Fonts/PingFang.ttc'  # 苹方（如果宋体没有）

prop_cn_large = FontProperties(fname=simsun_path, size=24)  # 用于坐标轴标签
prop_cn_small = FontProperties(fname=simsun_path, size=20)  # 用于图例、刻度

# 如果找不到宋体，可以用以下命令查看系统可用字体:
# from matplotlib import font_manager
# for font in font_manager.fontManager.ttflist:
#     if 'Song' in font.name or 'SimSun' in font.name or '宋' in font.name:
#         print(f"{font.name}: {font.fname}")

width = 0.8
color1 = '#91CBFC'  # 有点亮的浅蓝
color3 = '#729DC7'  # 深一点的浅蓝

# 读取CSV文件
df = pd.read_csv("CoMP_overhead.csv")
df['CoMP Stage1'] = df['CoMP Stage1'].str.rstrip('%').astype(float)
df['CoMP Stage2'] = df['CoMP Stage2'].str.rstrip('%').astype(float)
df['Train'] = 100 - (df['CoMP Stage1'] + df['CoMP Stage2'])
models = df['Model'].tolist()

plt.figure(figsize=(10, 6))

# 绘制堆叠条形图
plt.bar(models, df['CoMP Stage1'], bottom=df['Train'] + df['CoMP Stage2'], 
        color='#FFA543', width=width*0.82, label='CoMP 阶段一')
plt.bar(models, df['CoMP Stage2'], bottom=df['Train'], 
        color='#FF7F0E', width=width*0.82, label='CoMP 阶段二')
plt.bar(models, df['Train'], color=color3, width=width*0.82, label='后训练')

# 设置纵坐标刻度
plt.yticks([0, 25, 50, 75, 100])
plt.ylim(0, 100)

# 设置标题和标签 - 中文用宋体
plt.xlabel('模型', fontproperties=prop_cn_large)
plt.ylabel('开销占比 (%)', fontproperties=prop_cn_large)

# 移动图例位置 - 中文用宋体
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), 
           prop=prop_cn_small, ncol=3)

# 设置坐标轴字体
# X轴刻度：如果 model 名称是英文，用 Times；如果是中文，用宋体
# 假设 models 是英文（如 GPT, LLaMA），否则改为 prop_cn_small
plt.xticks(fontproperties=prop_en)  
plt.yticks([0, 25, 50, 75, 100], fontproperties=prop_en)

# 设置网格线
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("evaD_breakdown_CN.pdf", dpi=300, bbox_inches='tight')
plt.close()