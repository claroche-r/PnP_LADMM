import torch
import torch.nn as nn
import tqdm 
from models.pnp_blocks import upsample, o_leary_batch, transpose_o_leary_batch
from models.pnp_blocks import ResUNet as net
import time
import numpy as np


class PnP_approx_ADMM():
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda'
        self.lamb = self.opt['lamb']
        self.sigma_d = torch.FloatTensor([[[[self.opt['sigma_d']]]]]).to(self.device)
        self.beta = self.lamb / (self.sigma_d ** 2)
        self.n_iter = self.opt['n_iter']
        self.eta = self.opt['eta']
        self.load_denoiser()

    def get_hyperparams(self):
        return self.lamb, self.sigma_d.cpu().item(), None

    def load_denoiser(self):
        self.denoiser = net(in_nc=4, out_nc=3,   nc=[64, 128, 256, 512],
                            nb=4, act_mode='R',  downsample_mode='strideconv',
                            upsample_mode='convtranspose')
        self.denoiser.load_state_dict(torch.load(self.opt['path_denoiser']))
        self.denoiser = self.denoiser.to(self.device)
        
    def approx_prox_data(self, kmap, basis, y, x, sigma, x_0):
        # Init
        alpha = self.beta * (sigma ** 2)
        b = transpose_o_leary_batch(y, kmap, basis) +  x * alpha
        
        def A(x, kmap, basis, alpha):
            return transpose_o_leary_batch(o_leary_batch(x, kmap, basis), kmap, basis) + x * alpha
        
        x_k = torch.zeros_like(x)
        r = b.clone()
        p_k = b.clone()
        rsold = (r * r).sum(dim=(1,2,3))

        for i in range(100):
            Ap = A(p_k, kmap, basis, alpha)
            a_k = rsold / (p_k * Ap).sum(dim=(1,2,3))
            x_k = x_k + p_k * a_k[:, None, None, None]
            r = r - Ap * a_k[:, None, None, None]
            rsnew = (r * r).sum(dim=(1,2,3))
            if torch.sqrt(rsnew).max() <= 1e-2:
                break
            else:
                beta = (rsnew / rsold)
                p_k = r +  p_k  * beta[:, None, None, None]
                rsold = rsnew
        return x_k

    def feed_data(self, data):
        self.y = data['L'].to(self.device)
        self.sf = data['sf'][0]
        self.sigma = data['sigma'].to(self.device)
        self.kmap = data['kmap'].to(self.device)
        self.basis = data['basis'].to(self.device)
        try:
            self.ref = data['H']
        except:
            pass
        
    def run_iter(self, x, z, u, y, sigma, kmap, basis, sf):
        # --------------------------------
        # step 1, denoiser
        # --------------------------------
        x = self.denoiser(torch.cat((z - u, self.sigma_d.repeat(z.shape[0],1,z.shape[2], z.shape[3])), dim=1))


        # --------------------------------
        # step 2, data term gradient descent
        # --------------------------------
        z = self.approx_prox_data(kmap, basis, y, x + u, sigma, z)

        # --------------------------------
        # step 3, residuals
        # --------------------------------
        u = u + (x - z)
        
        return x, z, u
    
    def init_pnp(self, y, sf):
        upsampler = nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=False)
        x = upsampler(y)
        z = x
        u = x - z
        
        return x, z, u
    
    def run(self):
        with torch.no_grad():
            x, z, u = self.init_pnp(self.y, self.sf)
            x_list = [x]
            time_list = [0]

            for _ in range(self.n_iter):
                torch.cuda.synchronize()
                t1 = time.time()
                x, z, u = self.run_iter(x, z, u, self.y, self.sigma, self.kmap, self.basis, self.sf)
                torch.cuda.synchronize()
                t2 = time.time()

                x_list.append(x.cpu())
                time_list.append(t2-t1)
            
        return x.cpu(), x_list, time_list


    def fit_params(self):
        lamb_grid = [0.5, 1, 3, 5]
        sig_grid = [10/255, 20/255, 40/255, 60/255]
        current_mse = np.inf
        current_params = (None,None)

        # Grid search
        for sig in tqdm.tqdm(sig_grid, desc='Sigma loop'):
            sigD = torch.FloatTensor([[[[sig]]]]).to(self.device)
            self.sigma_d = sigD
            for lamb in tqdm.tqdm(lamb_grid, desc='Lambda loop'):
                self.beta = lamb / sigD**2
                self.lamb = lamb

                est, _, _ = self.run()
                mse = ((est - self.ref)**2)[...,17:-17,17:-17].mean().item()
                #mse=0

                if mse <= current_mse:
                    current_mse = mse
                    current_params = (sigD, lamb)

        # Update model with optimal params
        self.sigma_d  = current_params[0]
        self.lamb = current_params[1]
        self.beta = self.lamb / self.sigma_d ** 2

        print(60 * '-')
        print('Grid search completed!')
        print(60 * '-')
