from IPython.display import clear_output
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim,compare_psnr, compare_mse

from pathlib import Path
from dataset import *


def train_abs(n_epoch,n_train,n_batch,alpha,lr_u,n_steps,U_range,dataset,x_train):
    device_id = 0
    torch.cuda.set_device(device_id)
    
    
    x_train = np.expand_dims(x_train, axis=3)
    _, height, width, nc = x_train.shape
    
    zeropad = nn.ZeroPad2d(height//2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    dirs = f'./output_abs/20200202_{dataset}_{n_epoch}_{n_train//n_batch}_{n_batch}_{alpha}_{lr_u}_{n_steps}_{U_range}/'
    disk_dir = Path(dirs)
    disk_dir.mkdir(parents=True, exist_ok=True)


    x_train = x_train[:n_train,:,:,:].reshape(-1,nc,height,width)

    seed = 1000
    np.random.seed(seed)
    u = np.random.normal(0, 1, size=(nc,height,width)) # mu, sigma

    if U_range:
        u = u - np.min(u)
        u = u / np.max(u)
        u = u * U_range[1]

    u_iter=0
    np.save(disk_dir / f'u_{u_iter}', u)

    N_iter = np.int(np.ceil(n_train/np.float(n_batch)))
    idx = np.arange(x_train.shape[0])
    loss_per_epoch = []

    torch.autograd.set_detect_anomaly(True)
    eps_tensor = torch.cuda.FloatTensor([1e-15])    
    pi_tensor = torch.cuda.FloatTensor([np.pi])

    best_epoch = 0
    best_loss = float('inf')
    for epoch in range (n_epoch):
        print(epoch)
        x_train_rec = np.zeros_like(x_train) # reconstruction of x_train

        loss_epoch = []
        epoch_idx = idx
        np.random.shuffle(epoch_idx)

        for iters in range(N_iter):

            x = x_train[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_train])],:,:,:]
            x_gt = torch.cuda.FloatTensor(x).view(-1, nc, height, width).cuda()
            uk = torch.autograd.Variable(torch.cuda.FloatTensor(u).view(-1,nc,height,width),requires_grad=True)

            # z = x + u
            z = zeropad(x_gt + uk)
            dummy_zeros = torch.zeros_like(z).cuda()
            z_complex = torch.cat((z.unsqueeze(4), dummy_zeros.unsqueeze(4)), 4)

            Fz = torch.fft(z_complex, 2, normalized=True)
            # y = |F(x+u)| = |Fz|
            y = torch.norm(Fz, dim=4)
            y_dual = torch.cat((y.unsqueeze(4), y.unsqueeze(4)), 4)

            x_est = x_train_rec[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_train])],:,:,:]
            x_est = torch.cuda.FloatTensor(x_est).cuda()
            
            loss_pr=[]
            meas_loss_pr=[]
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
                
                loss_pr.append(np.mean((x-x_est.cpu().detach().numpy())**2))
                meas_loss_pr.append(height*2*width*2*np.mean((y.cpu().detach().numpy().reshape(-1,2*height,2*width)-
                    np.abs(np.fft.fft2(z_est.cpu().detach().numpy().reshape(-1,2*height,2*width), norm="ortho")))**2))
            
            x_train_rec[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_train])],:,:,:] = x_est.cpu().detach().numpy()


            # update u
            loss_u = (x_gt - x_est).pow(2).mean() * height * width
            loss_epoch.append(loss_u.item())
            loss_u.backward()
            with torch.no_grad():
                u_grad = uk.grad.data.cuda()
                new_uk = uk - lr_u * u_grad
                if U_range:
                    new_uk = torch.clamp(new_uk, U_range[0], U_range[1])
                uk = new_uk

        # save u at every epoch_epoch
        u = uk.cpu().detach().numpy()
        u_iter = u_iter + 1
        np.save(disk_dir / f'u_{u_iter}', u)


        mean_loss = np.array(loss_epoch).mean()
        loss_per_epoch.append(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = u_iter
        print(f'best so far {best_epoch}' )

        if epoch % 10 == 9:
            clear_output()
        # plot
        plt.figure()
        plt.semilogy(np.array(loss_per_epoch).flatten())
        plt.title(f'loss per epoch {loss_per_epoch[-1]}')
        plt.show()

        plt.figure()
        plt.imshow(u[0][0]);plt.gray()
        plt.title(f'u [{u.min()}, {u.max()}]')
        plt.show()

        mse = np.mean((x_train_rec-x_train)**2)
        psnr = 20*np.log10((np.max(x_train)-np.min(x_train))/np.sqrt(mse))
        print(f'psnr {psnr}')
        print(f'mse {mse}')
        psnr_list = [compare_psnr(x_train[i,0,:,:],x_train_rec[i,0,:,:]) for i in range(n_train)]

        

        n = np.min([100,n_train])
        figset = range(0,n)
        plt.figure(figsize=(n*2, 4))
        for i in range(n):
            # display original
            if nc==1:
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(x_train[figset[i]].reshape(height, width))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            
                # display reconstruction
                ax = plt.subplot(2, n, i + 1 +n)
                plt.imshow(x_train_rec[figset[i]].reshape(height, width))
                plt.title(f'{psnr_list[i]:.2f}')
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            elif nc==3:
                ax = plt.subplot(2, n, i + 1)
                temp=x_train[figset[i]]
                temp1=np.zeros((height, width,nc))
                for chan in range (0,nc):
                    temp1[:,:,chan]=temp[chan,:,:]
                plt.imshow(temp1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, n, i + 1 +n)
                temp=x_train_rec[figset[i]]
                temp1=np.zeros((height, width,nc))
                for chan in range (0,nc):
                    temp1[:,:,chan]=temp[chan,:,:]
                plt.imshow(temp1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.show()


    # save the last u
    u = uk.cpu().detach().numpy().reshape(nc,height,width)
    np.save(disk_dir / f'u_{dataset}_{n_train}',u)
    print(f'best_epoch {best_epoch}' )
