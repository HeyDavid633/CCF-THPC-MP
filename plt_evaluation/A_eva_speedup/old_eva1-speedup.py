# 7.10 对于eva1 speedup-bar的作图 
# 
# 字体也需要是我所要求的字体
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter, FixedFormatter

# 设置全局字体为自定义字体，增加字号
font_path = '../Times2.ttf'
fontprop = fm.FontProperties(fname=font_path, size=20)
# plt.rcParams['font.size'] = 18

df = pd.read_csv('./A100_eva1_speedup.csv')

# 创建一个大的图表，其中包含六个子图
fig, axs = plt.subplots(1, 6, figsize=(30, 5))
axs = axs.flatten()

# 颜色定义
colors = ['#9DC3E6', '#A9D18E', '#FFA543']

# 定义Y轴刻度
yticks = [0.5, 1.0, 1.5, 2.0]


for ax in axs:
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)  # 只显示y轴网格线
    
    
for i, row in df.iterrows():
    fp32_speed = row['FP32']
    amp_speed = row['AMP']
    emp_speed = row['EMP']
    
    ratios = [fp32_speed / fp32_speed, fp32_speed / amp_speed, fp32_speed / emp_speed]
    
    # 绘制柱状图
    axs[i].bar(['FP32', 'AMP', 'EMP'], ratios, color=colors, edgecolor='black', width=0.7)
    
    # 在柱状图上添加数值
    for j, ratio in enumerate(ratios):
        axs[i].text(j, ratio + 0.01, f'{ratio:.2f}x', ha='center', va='bottom', fontproperties=fontprop)

    # 添加子图标题，放置于子图下方
    axs[i].text(0.5, -0.15, f'({chr(97+i)}) {row["Net"]}', ha='center', va='top', transform=axs[i].transAxes, fontproperties=fontprop)
    
    axs[i].set_ylim(bottom=0)
    axs[i].set_yticks(yticks)
    axs[i].set_yticklabels(yticks, fontproperties=fontprop)  # 应用y轴字体 
    axs[i].set_xticklabels(['FP32', 'AMP', 'EMP'], fontproperties=fontprop) 
    axs[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    axs[i].set_title('') # 去除默认的title，因为我们将使用text来显示标题
    
    
    plt.xticks(fontproperties=fontprop)

# 调整子图布局
plt.tight_layout(rect=[0, 0, 1, 1])

plt.subplots_adjust(wspace=0.18) # 拉开子图间距

plt.savefig("eva1_A100_Speedup.pdf")

