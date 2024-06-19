import torch

def torch_cuda_active():
    if torch.cuda.is_available():
        print('PyTorch version\t:', torch.__version__)
        print('CUDA version\t:', torch.version.cuda)
        print('GPU\t\t:', torch.cuda.get_device_name(), '\n')
        return torch.device('cuda')
    else:
        print('CUDA is not available!')
        return torch.device('cpu')
