# 6.18  delata loss的统计 errorbar  
#
# 这里的 errorbar并不是真正的（定义中的errorbar应该是标准差作为上下限）
# 叫做垂直线区间更合适 
# 从epoch后的 第2个delta loss开始算，因为第一个deltaloss = 0
# 
# python demo_delta_loss_errorbar.py A100 vgg16
# 
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

cav_file_path = '../log_4080/'
batch_start = 20

def read_csv(file_path_fp32, file_path_amp):
    with open(file_path_fp32, 'r') as csvfile_fp32, open(file_path_amp, 'r') as csvfile_amp:
        reader_fp32 = csv.reader(csvfile_fp32)
        reader_amp = csv.reader(csvfile_amp)
        
        data_fp32 = [[abs(float(value)) for value in row[batch_start:]] for row in reader_fp32]
        data_amp = [[abs(float(value)) for value in row[batch_start:]] for row in reader_amp]
    
    return data_fp32, data_amp


def plot_vertical_lines_with_points(data_fp32, data_amp, output_image):
    bias_amp = 0.35  
    epochs = range(1, len(data_fp32) + 1)  # 假设两个数据集长度相同
    epochs2 = [e + bias_amp for e in epochs]
    
    min_losses_fp32 = [min(row) for row in data_fp32]
    max_losses_fp32 = [max(row) for row in data_fp32]
    avg_losses_fp32 = [np.mean(row) for row in data_fp32]
    
    min_losses_amp = [min(row) for row in data_amp]
    max_losses_amp = [max(row) for row in data_amp]
    avg_losses_amp = [np.mean(row) for row in data_amp]


    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制fp32数据的最大最小值区间
    for i in range(len(data_fp32)):
        ax.vlines(x=epochs[i], ymin=min_losses_fp32[i], ymax=max_losses_fp32[i], colors='blue', alpha=0.5, linewidth=2, label='FP32 Range' if i == 0 else "")
    
    # 绘制amp数据的最大最小值区间
    for i in range(len(data_amp)):
        ax.vlines(x=epochs2[i], ymin=min_losses_amp[i], ymax=max_losses_amp[i], colors='red', alpha=0.5, linewidth=2, label='AMP Range' if i == 0 else "")

    # 绘制fp32平均值曲线
    ax.plot(epochs, avg_losses_fp32, color='blue', linestyle='-', linewidth=2, label='FP32 Average')
    
    # 绘制amp平均值曲线
    ax.plot(epochs2, avg_losses_amp, color='red', linestyle='-', linewidth=2, label='AMP Average')
    
    # 新增计算差值序列
    diff_avg_losses = [amp - fp32 for amp, fp32 in zip(avg_losses_amp, avg_losses_fp32)]

    ax2 = ax.twinx()  # 创建第二个y轴
    ax2.plot(epochs, diff_avg_losses, color='green', linestyle='-', linewidth=2, label='AMP - FP32 Difference')
    ax2.set_ylabel('Difference (AMP - FP32)', color='black')  # 设置右侧y轴标签
    ax2.tick_params(axis='y', labelcolor='black')  # 设置右侧y轴颜色

    # 显式设置右边Y轴的刻度和范围
    ax2.set_yticks(np.arange(-0.0025, 0.003, 0.0005))  # 设置期望的刻度点
    ax2.set_ylim(-0.0025, 0.003)  # 强制设置Y轴的上下限，确保-0.01和0.01都在范围内
    

    # 更新图例位置，确保不遮挡数据
    handles, labels = [], []
    for ax_ in [ax, ax2]:
        for h, l in zip(*ax_.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)

    
    plt.title('Epochs with Vertical Ranges and Average Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute Loss Delta')
    plt.xlim(left=0)  
    # plt.ylim(bottom=0)  
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('platfrom', type=str, help='platfrom')
    parser.add_argument('net', type=str, help='net')
    args = parser.parse_args()
    
    platform = args.platfrom
    net = args.net
    
    fp32_filename = cav_file_path + net + '_fp32_100_delta_loss.csv'
    amp_filename = cav_file_path + net + '_amp_100_delta_loss.csv'
    save_pdf_name = platform + '_' + net + '_delta_loss_errorbar.pdf'
    
    data_fp32, data_amp = read_csv(fp32_filename, amp_filename)
    plot_vertical_lines_with_points(data_fp32, data_amp, save_pdf_name)
