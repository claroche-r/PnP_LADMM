import torch
import torch.nn as nn
from torch.nn import functional as F
from models.pnp_blocks import upsample, o_leary_batch, transpose_o_leary_batch
import time


def gradient_v2(img):

    G_x = torch.zeros_like(img).to(img.device)
    G_y = torch.zeros_like(img).to(img.device)
    C = img.shape[1]

    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).to(img.device)

    a = a.view((1, 1, 3, 3))

    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).to(img.device)

    b = b.view((1, 1, 3, 3))

    for c in range(C):
        G_x[:,c:c+1,:,:] = F.conv2d(img[:,c:c+1,:,:], a, padding=1)

        G_y[:,c:c+1,:,:]  = F.conv2d(img[:,c:c+1,:,:], b, padding=1)

    return G_y, G_x


def normalised_gradient_divergence(F):
    """ compute the divergence of n-D scalar field `F` """
    gy, gx = gradient_v2(F)
    eps = 1e-9
    ngy = gy.clone().to(F.device)
    ngx = gx.clone().to(F.device)
    norm = torch.sqrt(gx**2 + gy**2)
    ngy[norm < eps]=  eps
    ngx[norm < eps] = eps
    ngy[norm >= eps] = torch.div(gy[norm >= eps], norm[norm >= eps])
    ngx[norm >= eps] = torch.div(gx[norm >= eps], norm[norm >= eps])
    gxy, gxx = gradient_v2(ngx)
    gyy, gyx = gradient_v2(ngy)
    return gxx + gyy


class RichardsonLucy():
    def __init__(self, opt):
        self.n_iter = opt['n_iter']
        self.epsilon = 1e-6
        self.reg_factor = 1e-3
        self.device = 'cuda' if opt['gpu_id'] else 'cpu'

    def get_hyperparams(self):
        return None, None, None

    def feed_data(self, data):
        self.y = data['L'].to(self.device)
        self.kmap = data['kmap'].to(self.device)
        self.basis = data['basis'].to(self.device)
        try:
            self.ref = data['H']
        except:
            pass
    
    def run_iter(self, x, y, kmap, basis):
        x_reblurred = o_leary_batch(x, kmap, basis)
        relative_blur = torch.div(y, x_reblurred + self.epsilon)
        error_estimate = transpose_o_leary_batch(relative_blur, kmap, basis)
        x = x * error_estimate
        #J_reg_grad = self.reg_factor * normalised_gradient_divergence(x)
        #x = x * (1.0/(1-J_reg_grad))
        return x
    
    def run(self):
        with torch.no_grad():
            x = self.y
            x_list = [x]
            time_list = [0]

            for _ in range(self.n_iter):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                x = self.run_iter(x, self.y, self.kmap, self.basis)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                
                x = x.clamp(0,1)

                x_list.append(x.cpu())
                time_list.append(t2-t1)
        
        return x.cpu(), x_list, time_list

    def fit_params(self):
        pass
