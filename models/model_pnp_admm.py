import torch
import torch.nn as nn
import utils.utils_pnp as pnp
import time
import tqdm 
import numpy as np
from models.pnp_blocks import ResUNet as net


class PnP_ADMM():
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda'
        self.lamb = self.opt['lamb']
        self.sigma_d = torch.FloatTensor([[[[self.opt['sigma_d']]]]]).to(self.device)
        self.mu = self.lamb / self.sigma_d ** 2
        self.n_iter = self.opt['n_iter']
        self.load_denoiser()
    
    def get_hyperparams(self):
        return self.lamb, self.sigma_d, None

    def load_denoiser(self):
        self.denoiser = net(in_nc=4, out_nc=3,   nc=[64, 128, 256, 512],
                            nb=4, act_mode='R',  downsample_mode='strideconv',
                            upsample_mode='convtranspose')
        self.denoiser.load_state_dict(torch.load(self.opt['path_denoiser']))
        self.denoiser = self.denoiser.to('cuda')

    
    def feed_data(self, data):
        self.y = data['L'].to(self.device)
        self.sf = data['sf'][0]
        self.sigma = data['sigma'].to(self.device)
        self.k = data['basis'].to(self.device)
        try:
            self.ref = data['H']
        except:
            pass

    def run_iter(self, FB, FBC, F2B, FBFy, x, z, u, sigma, k, sf):
        # --------------------------------
        # step 1, denoiser
        # --------------------------------
        x = self.denoiser(torch.cat((z - u, self.sigma_d.repeat(z.shape[0],1,z.shape[2], z.shape[3])), dim=1))

        # --------------------------------
        # step 2, data term
        # --------------------------------
        z = pnp.data_solution((x + u), FB, FBC, F2B, FBFy, sigma**2 * self.mu, sf)

        # --------------------------------
        # step 3, residuals
        # --------------------------------
        u = u + (x - z)
        
        return x, z, u
    
    def init_pnp(self, y, k, sf):
        upsampler = nn.Upsample(scale_factor=sf, mode='bilinear')
        x = upsampler(y)
        z = x
        u = x - z
        FB, FBC, F2B, FBFy = pnp.pre_calculate(y, k, sf)
        
        return x, z, u, FB, FBC, F2B, FBFy
    
    def run(self):
        with torch.no_grad():
            x, z, u, FB, FBC, F2B, FBFy = self.init_pnp(self.y, self.k, self.sf)
            x_list = [x]
            time_list = [0]

            for _ in range(self.n_iter):
                t1 = time.time()
                x, z, u = self.run_iter(FB, FBC, F2B, FBFy, x, z, u, self.sigma, self.k, self.sf)
                t2 = time.time()

                x_list.append(x.cpu())
                time_list.append(t2-t1)
            
        return x.cpu(), x_list, time_list

    def fit_params(self):
        lamb_grid = [1, 3, 5, 7, 9]
        sig_grid = [5/255, 10/255, 20/255, 40/255, 60/255]
        current_mse = np.inf
        current_params = (None,None)

        # Grid search
        for sig in tqdm.tqdm(sig_grid, desc='Sigma loop'):
            sigD = torch.FloatTensor([[[[sig]]]]).to('cuda')
            self.sigma_d = sigD
            for lamb in tqdm.tqdm(lamb_grid, desc='Lambda loop'):
                self.mu = lamb / sigD**2
                self.lamb = lamb

                est, _, _ = self.run()
                mse = ((est - self.ref)**2)[...,17:-17,17:-17].mean().item()

                if mse <= current_mse:
                    current_mse = mse
                    current_params = (sigD, lamb)

        # Update model with optimal params
        self.sigma_d  = current_params[0]
        self.lamb = current_params[1]
        self.mu = self.lamb / self.sigma_d ** 2

        print(60 * '-')
        print('Grid search completed!')
        print(60 * '-')