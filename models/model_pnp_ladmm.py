import torch
import torch.nn as nn
from models.network_nimbusr import upsample, o_leary_batch, transpose_o_leary_batch
import time
import numpy as np
import tqdm 
from models.pnp_blocks import DataNetDeblur as data_net
from models.pnp_blocks import ResUNet as net


class PnP_linearized_ADMM():
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda'
        self.lamb = self.opt['lamb']
        self.sigma_d = torch.FloatTensor([[[[self.opt['sigma_d']]]]]).to(self.device)
        self.Lx = self.opt['Lx']
        self.beta = self.lamb / (self.sigma_d ** 2)
        self.n_iter = self.opt['n_iter']
        self.load_denoiser()
        self.prox_data = data_net().to(self.device)
    
    def get_hyperparams(self):
        return self.lamb, self.sigma_d.cpu().item(), self.Lx.cpu().item()
    
    def load_denoiser(self):
        self.denoiser = net(in_nc=4, out_nc=3,   nc=[64, 128, 256, 512],
                            nb=4, act_mode='R',  downsample_mode='strideconv',
                            upsample_mode='convtranspose')
        self.denoiser.load_state_dict(torch.load(self.opt['path_denoiser']))
        self.denoiser = self.denoiser.to(self.device)
    
    def feed_data(self, data):
        self.y = data['L'].to(self.device)
        self.sigma = data['sigma'].to(self.device)
        self.kmap = data['kmap'].to(self.device)
        self.sf = data['sf'][0]
        self.basis = data['basis'].to(self.device)
        try:
            self.ref = data['H']
        except:
            pass
        
    def run_iter(self, x, z, u, h, STy, kmap, basis, alpha, sigma_d_big, gamma):
        # --------------------------------
        # step 1, denoiser
        # --------------------------------
        i = x - gamma * transpose_o_leary_batch(h - z + u, kmap, basis)
        x = self.denoiser(torch.cat((i, sigma_d_big), dim=1))

        # --------------------------------
        # step 2, data term
        # --------------------------------
        h = o_leary_batch(x, kmap, basis)
        z = (STy + alpha * (h+u)) / (1 + alpha)

        # --------------------------------
        # step 3, residuals
        # --------------------------------
        u = u + (h - z)
        
        return x, z, u, h
    
    def init_pnp(self, y, kmap, basis, sf, init_x=None):
        if init_x == None:
            upsampler = nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=False)
            x = upsampler(y)
        else:
            x = init_x
        z = x
        h = o_leary_batch(x, kmap, basis)
        u = torch.zeros_like(z)
        STy = torch.zeros((y.shape[0], y.shape[1], y.shape[2]*sf, y.shape[3]*sf)).type_as(y)
        STy[..., ::sf, ::sf].copy_(y)
        
        return x, z, u, h, STy
    
    def run(self):
        with torch.no_grad():
            x, z, u, h, STy = self.init_pnp(self.y, self.kmap, self.basis, self.sf)
            x_list = [x]
            time_list = [0]
            gamma = (self.beta/self.Lx)
            sigma_d_big = self.sigma_d.repeat(z.shape[0],1,z.shape[2], z.shape[3])
            alpha = self.sigma ** 2 * self.beta

            for _ in range(self.n_iter):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                x, z, u, h = self.run_iter(x, z, u, h, STy, self.kmap, self.basis, alpha, sigma_d_big, gamma)
                torch.cuda.synchronize()
                t2 = time.perf_counter()

                x_list.append(x.cpu())
                time_list.append(t2-t1)
            
        return x.cpu(), x_list, time_list


    def fit_params(self):
        lamb_grid = [0.5, 1, 3, 5, 7]
        sig_grid = [5/255, 10/255, 20/255, 40/255]
        current_mse = np.inf
        current_params = (None, None, None)

        # Grid search
        for sig in tqdm.tqdm(sig_grid, desc='Sigma loop'):
            sigD = torch.FloatTensor([[[[sig]]]]).to(self.device)
            self.sigma_d = sigD
            for lamb1 in tqdm.tqdm(lamb_grid, desc='Lambda 1 loop'):
                for lamb2 in tqdm.tqdm([i for i in lamb_grid if i >= lamb1], desc='Lambda 2 loop'):
                    self.Lx = lamb2 / sigD**2
                    self.beta = lamb1 / sigD**2
                    self.lamb = lamb2

                    est, _, _ = self.run()
                    mse = ((est - self.ref)**2)[...,17:-17,17:-17].mean().item()
                    #mse = 0

                    if mse < current_mse:
                        current_mse = mse
                        current_params = (sigD, lamb1, lamb2)

        # Update model with optimal params
        self.sigma_d  = current_params[0]
        self.lamb = current_params[2]
        self.beta = current_params[1] / self.sigma_d ** 2
        self.Lx = self.lamb / self.sigma_d ** 2

        print(60 * '-')
        print('Grid search completed!')
        print(60 * '-')
