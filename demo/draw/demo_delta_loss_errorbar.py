# delata loss的统计曲线

import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # next(reader)  # 跳过表头
        data = [[abs(float(value)) for value in row[10:]] for row in reader]
    return data

def plot_vertical_lines_with_points(data, output_image='vertical_lines_amp_plot.pdf'):
    epochs = range(1, len(data) + 1)  # 横坐标为epoch序号
    min_losses = [min(row) for row in data]
    max_losses = [max(row) for row in data]
    avg_losses = [np.mean(row) for row in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制每行数据的最大最小值为区间的竖线
    for i in range(len(data)):
        ax.vlines(x=epochs[i], ymin=min_losses[i], ymax=max_losses[i], colors='blue', alpha=0.5, linewidth=2)

    # 在平均值处标记点
    ax.scatter(epochs, avg_losses, color='red', marker='o', label='Average Loss Delta')

    plt.title('Epochs with Vertical Range and Average Points (amp)')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute Loss Delta')
    plt.xlim(left=0)  # 确保x轴起始于0
    plt.ylim(bottom=0)  # 确保y轴起始于0
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_image)


if __name__ == "__main__":
    csv_file_path = '../log/vgg16_amp_10_delta_loss.csv'   # 确保路径正确指向你的CSV文件
    data = read_csv(csv_file_path)
    plot_vertical_lines_with_points(data)

