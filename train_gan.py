# 7.08 GAN网络训练脚本
# 
# 
# python train_gan.py -epoch 10 -precision emp
# 
# 
import argparse
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision
from conf import settings
from torchvision import transforms
from utils import torch_cuda_active, load_ImageNet

MNIST_dataset_path = '/workspace/CCF-THPC-MP/data'
each_epoch_time = []


def fp32_train(epoch):
    gan_discriminator.train()
    gan_generator.train()
    d_epoch_loss = 0.0
    g_epoch_loss = 0.0  
    tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch}', ncols=100)
    epoch_start_time = timeit.default_timer()
    
    for step, (img, _) in enumerate(tqdm_bar): # enumerate加序号
        
        img = img.to(device)
        batch_size = img.size(0) # 获取每一个批次的大小
        random_noise = torch.randn(batch_size, 100, device=device)  # 随机噪声的大小是size个
        gen_img = gan_generator(random_noise)  
        d_optim.zero_grad() 
        g_optim.zero_grad() 
        
        real_output = gan_discriminator(img)     
        d_real_loss = loss_function(real_output, torch.ones_like(real_output))      
        fake_output = gan_discriminator(gen_img.detach()) 
        d_fake_loss = loss_function(fake_output, torch.zeros_like(fake_output))      
        d_real_loss.backward() 
        d_fake_loss.backward() 
        d_optim.step() 
        
        fake_output = gan_discriminator(gen_img) 
        g_loss = loss_function(fake_output, torch.ones_like(fake_output))      
        g_loss.backward() 
        g_optim.step() 
        
        d_loss = d_real_loss + d_fake_loss # 判别器总的损失等于两个损失之和
        d_epoch_loss += d_loss
        g_epoch_loss += g_loss
        
        postfix = {'d_Loss': f"{d_epoch_loss / (step + 1):.4f}", 'g_Loss': f"{g_epoch_loss / (step + 1):.4f}"}
        tqdm_bar.set_postfix(**postfix)
        
    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time
    each_epoch_time.append(epoch_elapsed_time)

def amp_train(epoch):
    gan_discriminator.train()
    gan_generator.train()
    d_epoch_loss = 0.0
    g_epoch_loss = 0.0  
    tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch}', ncols=100)
    epoch_start_time = timeit.default_timer()
    
    for step, (img, _) in enumerate(tqdm_bar): # enumerate加序号
        
        img = img.to(device)
        batch_size = img.size(0) # 获取每一个批次的大小
        random_noise = torch.randn(batch_size, 100, device=device)  # 随机噪声的大小是size个
        gen_img = gan_generator(random_noise)  
        d_optim.zero_grad(set_to_none=True) 
        g_optim.zero_grad(set_to_none=True) 
        
        with autocast():
            real_output = gan_discriminator(img)     
            d_real_loss = loss_function(real_output, torch.ones_like(real_output))      
            fake_output = gan_discriminator(gen_img.detach()) 
            d_fake_loss = loss_function(fake_output, torch.zeros_like(fake_output))    
            fake_output = gan_discriminator(gen_img) 
            g_loss = loss_function(fake_output, torch.ones_like(fake_output))   
        
        d_loss = d_real_loss + d_fake_loss # 判别器总的损失等于两个损失之和   
        d_scaler.scale(d_loss).backward()  
        d_scaler.step(d_optim)  
        d_scaler.update() 
        
        g_scaler.scale(g_loss).backward()
        g_scaler.step(g_optim) 
        g_scaler.update()
        
        d_epoch_loss += d_loss
        g_epoch_loss += g_loss
        
        postfix = {'d_Loss': f"{d_epoch_loss / (step + 1):.4f}", 'g_Loss': f"{g_epoch_loss / (step + 1):.4f}"}
        tqdm_bar.set_postfix(**postfix)
        
    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time
    each_epoch_time.append(epoch_elapsed_time)

@torch.no_grad()
def eval_training(epoch=0):
    gan_discriminator.eval()
    total_correct = 0
    total_samples = 0
    
    for img, _ in val_dataloader:
        img = img.to(device)
        batch_size = img.size(0)
        
        # 真实图像
        real_output = gan_discriminator(img)
        real_pred = (real_output > 0.5).float()  # 判别器预测的真实图像是否为真
        total_correct += (real_pred == 1).sum().item() * 1.0
        
        # 生成图像
        random_noise = torch.randn(batch_size, 100, device=device)
        fake_images = gan_generator(random_noise)
        fake_output = gan_discriminator(fake_images)
        fake_pred = (fake_output < 0.5).float()  # 判别器预测的生成图像是否为假
        total_correct += (fake_pred == 0).sum().item() * 1.0
        
        total_samples += batch_size * 2  # 因为我们评估了真实和生成的图像
    
    if epoch % 5 == 0 or settings.EPOCH == epoch:
        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}], Accuracy: {accuracy:.4f}")
        
    return total_correct / total_samples

    
    
if __name__ == '__main__':
    torch.manual_seed(0)    #确保不同精度在进入权重相同  一般是0  1对于alexnet好收敛
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=5, help='epoch to train')
    parser.add_argument('-precision', type=str, default='fp32', help='use which precision: fp32|amp')
    parser.add_argument('-gpu', type=str, default=True, help='if use gpu')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for dataloader')
    args = parser.parse_args()
    settings.EPOCH = args.epoch
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    train_datasets = torchvision.datasets.MNIST(MNIST_dataset_path, train=True, transform=transform, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    val_datasets = torchvision.datasets.MNIST(MNIST_dataset_path, train=False, transform=transform, download=True)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False)
    
    # train_dataloader, val_dataloader, train_datasets, val_datasets = load_ImageNet(batch_size = args.batch_size, workers = 4)
    
    device = torch_cuda_active()

    if args.precision != 'emp':
        from models.gan import Generator, Discriminator
        gan_generator = Generator().to(device)
        gan_discriminator = Discriminator().to(device)
        
    else:
        from models.gan import Generator_emp, Discriminator_emp
        # model_layer_num = 6
        policy_precision_string = '000000'
        gan_generator = Generator_emp(policy_precision_string = policy_precision_string).to(device)
        gan_discriminator = Discriminator_emp(policy_precision_string = policy_precision_string).to(device)
    
    
    g_optim = torch.optim.Adam(gan_generator.parameters(), lr=0.0001)
    d_optim = torch.optim.Adam(gan_discriminator.parameters(), lr=0.0001)
    loss_function = nn.BCEWithLogitsLoss()   #  to get performance time
    # loss_function = nn.BCELoss()   # to get accuracy
    
    g_scaler = GradScaler(enabled=True)
    d_scaler = GradScaler(enabled=True)
    best_acc = 0.0
    
    train_start_time = timeit.default_timer()
    for epoch in range(1, settings.EPOCH + 1): 
        if args.precision == 'fp32' or 'emp':
            fp32_train(epoch)
        elif args.precision == 'amp':
            amp_train(epoch)
        
        acc = eval_training(epoch)
        
        if best_acc < acc:
            best_acc = acc
            continue
            
    train_end_time = timeit.default_timer()


    print('\n\nTraining summary', '-'*50,'\nGAN {} epoch, \tPrecision Policy: {}, \tTotal training time: {:.2f} min'.format(settings.EPOCH, args.precision, (train_end_time - train_start_time)/60))
    print('Each epoch average cost time: {:.4f} sec, \tFinal Accuracy: {:.4f}'.format(sum(each_epoch_time) / settings.EPOCH, best_acc))
    print('Max GPU memory: {:.2f} MB'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)))
    
    if settings.EPOCH <= 5: 
        summary_info_txt_filename = 'Log_Performance_GPU_memory.txt'
    else:
        summary_info_txt_filename = 'Log_Accuracy.txt'
        
    with open(summary_info_txt_filename, 'a') as f: 
        print('\n\nTraining summary', '-'*50,'\nGAN {} epoch, \tPrecision Policy: {}, \tTotal training time: {:.2f} min'.format(settings.EPOCH, args.precision, (train_end_time - train_start_time)/60), file=f)
        print('Each epoch average cost time: {:.2f} sec, \tFinal Accuracy: {:.4f}'.format(sum(each_epoch_time) / settings.EPOCH, best_acc), file=f)
        print('Max GPU memory: {:.2f} MB'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)), file=f)