import torch
import torch.nn as nn
import tqdm 
from models.pnp_blocks import upsample, o_leary_batch, transpose_o_leary_batch
from models.pnp_blocks import ResUNet as net
import time
import numpy as np


class PnP_FISTA():
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda'
        self.gamma = self.opt['eta']
        self.sigma_d = torch.FloatTensor([[[[self.opt['sigma_d']]]]]).to(self.device)
        self.n_iter = self.opt['n_iter']
        self.load_denoiser()

    def get_hyperparams(self):
        return self.gamma, self.sigma_d.cpu().item(), None

    def load_denoiser(self):
        self.denoiser = net(in_nc=4, out_nc=3,   nc=[64, 128, 256, 512],
                            nb=4, act_mode='R',  downsample_mode='strideconv',
                            upsample_mode='convtranspose')
        self.denoiser.load_state_dict(torch.load(self.opt['path_denoiser']))
        self.denoiser = self.denoiser.to(self.device)

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

    def grad_data_term(self, x, y, sigma, kmap, basis):
        res = o_leary_batch(x, kmap, basis) - y
        return transpose_o_leary_batch(res, kmap, basis)
        
    def run_iter(self, z, x, q, s, y, sigma, kmap, basis, sf):
        q_old = q
        x_old = x
        # --------------------------------
        # step 1, grad descent
        # --------------------------------
        z = s  - self.gamma * self.grad_data_term(s, y, sigma, kmap, basis)


        # --------------------------------
        # step 2, denoiser
        # --------------------------------
        x = self.denoiser(torch.cat((z, self.sigma_d.repeat(z.shape[0],1,z.shape[2], z.shape[3])), dim=1))


        # --------------------------------
        # step 3, q-update
        # --------------------------------
        q = 0.5 * (1 + torch.sqrt(1 + 4 * (q ** 2)))

        # --------------------------------
        # step 4, s-update
        # --------------------------------
        s = x + ((q_old - 1) / q) * (x - x_old)
        
        return z, x, q, s
    
    def init_pnp(self, y, sf):
        upsampler = nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=False)
        s = upsampler(y).to(self.device)
        z = s.to(self.device)
        x = s.to(self.device)
        q = torch.FloatTensor(1).to(self.device)
        
        return z, x, q, s
    
    def run(self):
        with torch.no_grad():
            z, x, q, s = self.init_pnp(self.y, self.sf)
            x_list = [x]
            time_list = [0]

            for _ in range(self.n_iter):
                t1 = time.time()
                z, x, q, s = self.run_iter(z, x, q, s, self.y, self.sigma, self.kmap, self.basis, self.sf)
                t2 = time.time()

                x_list.append(x.cpu())
                time_list.append(t2-t1)
            
        return x.cpu(), x_list, time_list


    def fit_params(self):
        gamma_grid = [0.1, 0.5, 0.9, 1.5, 1.9]
        sig_grid = [10/255, 20/255, 40/255, 60/255]
        current_mse = np.inf
        current_params = (None,None)

        # Grid search
        for sig in tqdm.tqdm(sig_grid, desc='Sigma loop'):
            sigD = torch.FloatTensor([[[[sig]]]]).to(self.device)
            self.sigma_d = sigD
            for gamma in tqdm.tqdm(gamma_grid, desc='Gamma loop'):
                self.gamma = gamma

                est, _, _ = self.run()
                mse = ((est - self.ref)**2)[...,17:-17,17:-17].mean().item()

                if mse <= current_mse:
                    current_mse = mse
                    current_params = (sigD, gamma)

        # Update model with optimal params
        self.sigma_d  = current_params[0]
        self.gamma = current_params[1]

        print(60 * '-')
        print('Grid search completed!')
        print(60 * '-')
