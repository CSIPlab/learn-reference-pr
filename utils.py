'''
GPU version
Author: Zikui Cai
'''
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)


def angle(x):
    a = np.angle(x)
    check_pos = a >= 0
    return check_pos*a + (1-check_pos)*(2*np.pi+a)

def np_sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def to_complex_tensor(x):
    # input [batch,h,w]
    batch = x.shape[0]
    m,n = x.shape[1],x.shape[2]
    x = torch.reshape(x,(batch,m,n,1))
    x = torch.cat((x, torch.zeros_like(x).to(dev)), dim=3)
    return x

def complex_mm(x,y):
    # x and y are [batch,h,w,2]
    # (a+bj) * (c+dj) = ac-bd + (ad+bc)j
    ac = x[:,:,:,0:1]*y[:,:,:,0:1]
    bd = x[:,:,:,1:]*y[:,:,:,1:]
    ad = x[:,:,:,0:1]*y[:,:,:,1:]
    bc = x[:,:,:,1:]*y[:,:,:,0:1]
    xy = torch.cat((ac-bd, ad+bc), dim=3)
    return xy

def zero_pad(x):
    # x is input signal, size (m x n x c)
    # F.pad - (c1, c2, n1, n2, m1, m2)
    #  (front, back, left, right, up, down)
    # pad around the input signal
    m,n = x.shape[0],x.shape[1]
    return F.pad(x,(0,0,n,n,m,m), mode='constant', value=0)

def zero_pad_right_down(x,m,n):
    # pad in 2 directions (right and down)
    return F.pad(x,(0,0,0,n,0,m), mode='constant', value=0)

def zero_pad_4sides(x,m,n):
    # pad in 2 directions (right and down)
    return F.pad(x,(0,0,n,n,m,m), mode='constant', value=0)

def center_crop(x):
    if len(x.shape) == 4:
        m3,n3 = x.shape[1],x.shape[2]
        m, n = m3//3, n3//3
        return x[:,m:2*m,n:2*n,0]
    elif len(x.shape) == 3:
        m3,n3 = x.shape[0],x.shape[1]
        m, n = m3//3, n3//3
        return x[m:2*m,n:2*n,0]
    elif len(x.shape) == 2:
        m3,n3 = x.shape[0],x.shape[1]
        m, n = m3//3, n3//3
        return x[m:2*m,n:2*n]

def make_real_mask(m,n):
    real_mask = torch.cat((torch.ones(m,n,1),torch.zeros(m,n,1)),dim=2)
    return real_mask
    
def make_imag_mask(m,n):
    imag_mask = torch.cat((torch.zeros(m,n,1),torch.ones(m,n,1)),dim=2)
    return imag_mask

def make_pad_mask(m,n):
    # m,n are the shape of the input image
    pad_mask = torch.ones([m,n,2])
    # pad_mask = torch.ones([corner_size,corner_size,2])
    # pad_mask = zero_pad_right_down(pad_mask,m-corner_size,n-corner_size)
    pad_mask = zero_pad(pad_mask)
    return pad_mask

def make_pad_mask_u_corner(corner_size,m,n):
    # m,n are the shape of the input image
    # pad_mask = torch.ones([m,n,2])
    pad_mask = torch.ones([corner_size,corner_size,2])
    pad_mask = zero_pad_right_down(pad_mask,m-corner_size,n-corner_size)
    pad_mask = zero_pad(pad_mask)
    return pad_mask

def make_pad_mask_u_corner_apart(corner_size,m,n):
    # m,n are the shape of the input image
    # pad_mask = torch.ones([m,n,2])
    pad_mask = torch.ones([corner_size,corner_size,2])
    pad_mask = zero_pad_right_down(pad_mask,3*m-corner_size,3*n-corner_size)
    return pad_mask

def make_pad_mask_u_center(center_size,m,n):
    # m,n are the shape of the input image
    # pad_mask = torch.ones([m,n,2])
    pad_mask = torch.ones([center_size,center_size,2])
    pad_mask = zero_pad_4sides(pad_mask,(m-center_size)//2,(n-center_size)//2)
    pad_mask = zero_pad(pad_mask)
    return pad_mask


A = lambda x : torch.fft(x,2,normalized=True)
B = lambda x : torch.fft(x,2,normalized=True)
Aconj = lambda x : torch.ifft(x,2,normalized=True)
Mag =  lambda x : to_complex_tensor(torch.norm(x,dim=3))
Mag2 = lambda x : to_complex_tensor(torch.pow(torch.norm(x,dim=3),2))


def load_u_trained(base_path,nth_iter,ku):
    disk_dir = Path(base_path)
    u = np.load(disk_dir / f"u_{nth_iter}_{ku}.npy")
    u = torch.from_numpy(u)
    return u


def prepare_u(x):
    m,n = x.shape[0],x.shape[1]
    x = torch.from_numpy(x)
    x = torch.reshape(x,(m,n,1))
    x = torch.cat((x, torch.zeros_like(x)), dim=2)
    x = F.pad(x,(0,0,n,n,m,m), mode='constant', value=0)
    return x


def init_constant_corner(c,m,n,M,N):
    # m,n is the corner size
    # M,N is the full size
    u = c*torch.ones([m,n,1])
    u = F.pad(u,(0,0,0,N-n,0,M-m), mode='constant', value=0)
    u = zero_pad(u)
    u = torch.cat((u, torch.zeros_like(u)), dim=2)
    return u

def load_batches(dataset,nth_iter,Batch):
    m,n = dataset[0].shape[0],dataset[0].shape[1]
    x_batch = torch.zeros([Batch,3*m,3*n,2])
    idx = 0
    for i in range(nth_iter*Batch,nth_iter*Batch+Batch):
        x = torch.from_numpy(np.expand_dims(dataset[i], axis=2))
        x = torch.cat((x, torch.zeros_like(x)), dim=2)
        x = zero_pad(x)
        x_batch[idx] = x
        idx += 1
    return x_batch.to(dev)


def load_batches_with_noise(dataset,nth_iter,Batch):
    m,n = dataset[0].shape[0],dataset[0].shape[1]
    x_batch = torch.zeros([Batch,3*m,3*n,2])
    idx = 0
    for i in range(nth_iter*Batch,nth_iter*Batch+Batch):
        noisy_data = dataset[i] + np.random.normal(0,0.01,dataset[i].shape)
        x = torch.from_numpy(np.expand_dims(noisy_data, axis=2))
        x = torch.cat((x, torch.zeros_like(x)), dim=2)
        x = zero_pad(x)
        x_batch[idx] = x
        idx += 1
    return x_batch.to(dev)


def load_batch_at_idx(dataset,idx):
    m,n = dataset[0].shape[0],dataset[0].shape[1]
    N_batch = len(idx)
    x_batch = torch.zeros([N_batch,3*m,3*n,2])
    for i in range(N_batch):
        x = torch.from_numpy(np.expand_dims(dataset[idx[i]], axis=2))
        x = torch.cat((x, torch.zeros_like(x)), dim=2)
        x = zero_pad(x)
        x_batch[i] = x
    return x_batch.to(dev)


def plot_dataset(dataset,nth_iter,batch_size):
    if batch_size <= 16:
        column = batch_size
        row = 1
    else:
        column = 16
        row = batch_size//16
    for r in range(row):
        fig, ax = plt.subplots(1, column,figsize=(20, 1))
        plt.gray()
        for c in range(column):
            i = r*column+c
            image = dataset[i+batch_size*nth_iter]
            title = f"{i+batch_size*nth_iter}"
            ax[c].set_title(title)
            ax[c].imshow(image)
        [axi.set_axis_off() for axi in ax.ravel()]
        plt.show()

from skimage.measure import compare_ssim,compare_psnr


def l2(x,y):
    if x.ptp() != 0:
        x_norm = (x-x.min())/(x.ptp())
    else:
        x_norm = x
    if y.ptp() != 0:
        y_norm = (y-y.min())/(y.ptp())
    else:
        y_norm = y
    loss = np.sqrt(np.mean(np.power(x_norm-y_norm, 2)))
    return loss

def compute_psnr(x, y):
    if x.ptp() != 0:
        x_norm = (x-x.min())/(x.ptp())
    else:
        x_norm = x
    if y.ptp() != 0:
        y_norm = (y-y.min())/(y.ptp())
    else:
        y_norm = y
    mse = np.mean(np.power(x_norm-y_norm,2))
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))


def SNR(image,noise_level):
    """
    Given signal and noise level, generate gaussian noise
    Return: noisy image
    """
    relu = lambda x : np.maximum(0, x)

    row,col= image.shape
    gauss = np.random.normal(0,0.1,(row,col))
    
    
    pow_signal = np.sum(np.square(image))
    pow_noise = np.sum(np.square(gauss))
    # gauss = relu(gauss)
    k = np.sqrt(pow_signal/pow_noise/np.power(10,noise_level/10))
    noise = k*gauss
    noise = relu(noise)
    pow_noise = np.sum(np.square(noise))
    snr = 10*np.log10(pow_signal/pow_noise)
    
    # SNR is frequently defined as the ratio of the signal power and the noise power
    #     plt.figure(figsize=(15,5))
    #     plt.subplot(1,3,1)
    #     plt.imshow(image)
    #     plt.subplot(1,3,2)
    #     plt.imshow(noise)
    #     plt.subplot(1,3,3)
    #     plt.imshow(image+noise)
    #     plt.show()
    
    return image+noise

def plot_recovery(x,x_train,nth_iter,batch_size):
    gt_data = load_batches(x_train,nth_iter,batch_size).cpu()
    if batch_size <= 16:
        column = batch_size
        row = 1
    else:
        column = 16
        row = batch_size//16
    for r in range(row):
        fig, ax = plt.subplots(1, column,figsize=(20, 1))
        for c in range(column):
            i = r*column+c
            image = center_crop(x[i])
            gt = center_crop(gt_data[i].numpy())
            loss_psnr = psnr(image,gt)
            title = f"${loss_psnr:.2f}$"
            ax[c].set_title(title)
            ax[c].imshow(image)
        [axi.set_axis_off() for axi in ax.ravel()]
        plt.show()


def plot_img_list(x_list,N_column=16,height_row=1):
    batch_size = len(x_list)
    if batch_size <= N_column:
        column = batch_size
        row = 1
    else:
        column = N_column
        row = np.ceil(batch_size/N_column).astype(np.int)
    fig, ax = plt.subplots(row, column,figsize=(20, height_row*row))
    plt.gray()
    for i in range(batch_size):
        if x_list[i].requires_grad:
            x = x_list[i].detach().numpy()
        else:
            x = x_list[i]
        if row == 1:
            ax[i].imshow(center_crop(x))
        else:
            ax[i//column,i%column].imshow(center_crop(x))
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()

def plot_img_with_title(imgs,titles):
    assert len(imgs) == len(titles),"len(imgs) and len(titles) don't match"
    fig, ax = plt.subplots(1, len(imgs),figsize=(20, 2))
    plt.gray()
    for i in range(len(imgs)):
        ax[i].imshow(imgs[i])
        ax[i].set_title(titles[i])
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()


print("Loaded util functions")