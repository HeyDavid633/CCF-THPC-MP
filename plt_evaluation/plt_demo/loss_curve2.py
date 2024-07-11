# 关于loss的绘图脚本
# fp32 / amp 的曲线
# 
# python loss_curve2.py 4080 alexnet
#
import matplotlib.pyplot as plt
import argparse
import csv


def plot_losses_from_csvs(platfrom, net):
    
    fp32_filename='../log/imagenet/' + net +'_fp32_50_loss.csv'
    amp_filename='../log/imagenet/' + net + '_amp_50_loss.csv'
    epochs, fp32_losses, amp_losses = [], [], []
    amp_accuracy = None
    amp_training_time = None
    fp32_accuracy = None
    fp32_training_time = None

    with open(fp32_filename, mode='r') as csvfile:
        next(csvfile) 
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                epochs.append(int(row[0]))
                fp32_losses.append(float(row[1]))
            elif len(row) == 1:
                if fp32_accuracy is None:
                    fp32_accuracy = float(row[0])
                else:
                    fp32_training_time = float(row[0])
    
    with open(amp_filename, mode='r') as csvfile:
        next(csvfile)  # 跳过标题行
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                amp_losses.append(float(row[1]))
            elif len(row) == 1:
                if amp_accuracy is None:
                    amp_accuracy = float(row[0])
                else:
                    amp_training_time = float(row[0])
                    
    

    assert len(fp32_losses) == len(amp_losses), "The number of epochs in both CSVs must match."

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fp32_losses, label='FP32 Training Loss', color='blue')
    plt.plot(epochs, amp_losses, label='AMP Training Loss', color='green')
    
    text_x_position = max(epochs) * 0.9
    plt.text(text_x_position, 5, f'AMP Accuracy: {amp_accuracy:.4f}\nAMP Training Time: {amp_training_time:.2f} min', horizontalalignment='right')
    plt.text(text_x_position, 4, f'FP32 Accuracy: {fp32_accuracy:.4f}\nFP32 Training Time: {fp32_training_time:.2f} min', horizontalalignment='right')

    plt.title(f'{platfrom} {net} Training Loss per Epoch (FP32 / AMP)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{platfrom}_{net}_imagenet_loss_curve.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('platfrom', type=str, help='platfrom')
    parser.add_argument('net', type=str, help='net')
    args = parser.parse_args()
    
    plot_losses_from_csvs(args.platfrom, args.net)
    
    
    
