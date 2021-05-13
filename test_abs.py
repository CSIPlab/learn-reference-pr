from IPython.display import clear_output
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error
from utils import compute_psnr

from dataset import *
from tqdm import tqdm


def test_abs(n_test,n_batch,n_steps,alpha,u,x_test):

    x_test = np.expand_dims(x_test, axis=3)
    _, height, width, nc = x_test.shape

    device_id = 0
    torch.cuda.set_device(device_id)

    zeropad = nn.ZeroPad2d(height//2)

    x_test = x_test[:n_test,:,:,:].reshape(-1,nc,height,width)

    N_iter = np.int(np.ceil(n_test/np.float(n_batch)))
    x_test_rec = np.zeros_like(x_test)

    eps_tensor = torch.cuda.FloatTensor([1e-15])
    epoch_idx = np.arange(n_test)

    pbar = tqdm(range(N_iter))
    for iters in pbar:

        x = x_test[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:]
        x_gt = torch.cuda.FloatTensor(x).view(-1, nc, height, width).cuda()
        uk = torch.cuda.FloatTensor(u).view(-1,nc,height,width)

        # z = x + u
        z = zeropad(x_gt + uk)
        dummy_zeros = torch.zeros_like(z).cuda()
        z_complex = torch.cat((z.unsqueeze(4), dummy_zeros.unsqueeze(4)), 4)

        Fz = torch.fft(z_complex, 2, normalized=True)
        # y = |F(x+u)| = |Fz|
        y = torch.norm(Fz, dim=4)
        y_dual = torch.cat((y.unsqueeze(4), y.unsqueeze(4)), 4)

        x_est = x_test_rec[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:]
        x_est = torch.cuda.FloatTensor(x_est).cuda()
        
        # image loss and measurement loss
        loss_x_pr=[]
        loss_y_pr=[]
        for kx in range(n_steps):
            
            z_est = zeropad(x_est + uk + eps_tensor)
            z_est_complex = torch.cat((z_est.unsqueeze(4), dummy_zeros.unsqueeze(4)), 4)
            Fz_est = torch.fft(z_est_complex,2, normalized=True)
            y_est = torch.norm(Fz_est,dim=4)
            y_est_dual = torch.cat((y_est.unsqueeze(4), y_est.unsqueeze(4)), 4)
            # angle Fz
            Fz_est_phase = Fz_est / (y_est_dual + eps_tensor)
            # update x
            x_grad_complex = torch.ifft( Fz_est - torch.mul(Fz_est_phase, y_dual), 2, normalized=True)
            x_grad = x_grad_complex[:,:, height//2:height//2+height, width//2:width//2+width, 0]
            x_est = x_est - alpha * x_grad
            x_est = torch.clamp(x_est, 0, 1)
            
            loss_x_pr.append(np.mean((x-x_est.cpu().detach().numpy())**2))
            loss_y_pr.append(height*2*width*2*np.mean((y.cpu().detach().numpy().reshape(-1,2*height,2*width)-
                np.abs(np.fft.fft2(z_est.cpu().detach().numpy().reshape(-1,2*height,2*width), norm="ortho")))**2))
        
        x_test_rec[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:] = x_est.cpu().detach().numpy()

    
    mse_list = [mean_squared_error(x_test[i,0,:,:],x_test_rec[i,0,:,:]) for i in range(n_test)]
    psnr_list = [compute_psnr(x_test[i,0,:,:],x_test_rec[i,0,:,:]) for i in range(n_test)]
    ssim_list = [structural_similarity(x_test[i,0,:,:],x_test_rec[i,0,:,:]) for i in range(n_test)]
    print(f'mse {np.mean(mse_list):.2f}')
    print(f'psnr {np.mean(psnr_list):.2f}')
    print(f'ssim {np.mean(ssim_list):.2f}')

    mse = np.mean((x_test_rec-x_test)**2)
    psnr = 20*np.log10((np.max(x_test)-np.min(x_test))/np.sqrt(mse))
    print(f'mean mse {mse:.2f}')
    print(f'psnr of mean {psnr:.2f}')
    print(f'psnr of mean (mean of psnr) {psnr:.2f}({np.mean(psnr_list):.2f})')

    return x_test_rec,mse_list,psnr_list,ssim_list


def plot_test(x_test_rec,x_test,mse_list,psnr_list,ssim_list,n_test,plot_n = 100):
    
    _, height, width = x_test.shape
    nc = 1
    
    plt.figure(figsize=(20,4))
    plt.subplot(131)
    plt.hist(mse_list,100)
    plt.title(f"MSE \n mean {np.mean(mse_list):.2f} std {np.std(mse_list):.2f}")
    plt.subplot(132)
    plt.hist(psnr_list,100)
    plt.title(f"PSNR \n mean {np.mean(psnr_list):.2f} std {np.std(psnr_list):.2f}")
    plt.subplot(133)
    plt.hist(ssim_list,100)
    plt.title(f"SSIM \n mean {np.mean(ssim_list):.2f} std {np.std(ssim_list):.2f}")
    plt.show()
    
    
    n = np.min([plot_n,n_test])
    figset = range(0,n)
    plt.figure(figsize=(n*2, 4))
    plt.gray()

    for i in range(n):
        # display original
        if nc==1:
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[figset[i]].reshape(height, width))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 +n)
            plt.imshow(x_test_rec[figset[i]].reshape(height, width))
            plt.title(f'{psnr_list[i]:.2f}')
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        elif nc==3:
            ax = plt.subplot(2, n, i + 1)
            temp=x_test[figset[i]]
            temp1=np.zeros((height, width,nc))
            for chan in range (0,nc):
                temp1[:,:,chan]=temp[chan,:,:]
            plt.imshow(temp1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 +n)
            temp=x_test_rec[figset[i]]
            temp1=np.zeros((height, width,nc))
            for chan in range (0,nc):
                temp1[:,:,chan]=temp[chan,:,:]
            plt.imshow(temp1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
