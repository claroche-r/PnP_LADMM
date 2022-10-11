import argparse
import pandas as pd
import numpy as np
import os
import tqdm 
import random
import torch
from torch.utils.data import DataLoader
from operator import add


import pickle

from models.select_model import define_Model
from data.dataset_pickle import Dataset as data_pickle
from data.dataset_multiblur import Dataset as data_multiblur
from utils.utils_pnp import get_metrics
import utils.utils_image as util

parser = argparse.ArgumentParser(description='Options for the plug-and-play model')
parser.add_argument('--input_df_path', type=str, default=None,
                    help='Path of the dataframe with the results')
parser.add_argument('--save_df_path', type=str, default=None,
                    help='Path of the dataframe with the results')
parser.add_argument('--path_denoiser', type=str, default='model_zoo/drunet_color.pth',
                    help='Path of the weights for DRUNet')
parser.add_argument('--model', type=str, default='linearized_ADMM',
                    help='Model to do the inference, choose in ["admm", "approximate_admm" , "linearized_admm", "richardson_lucy"]')                 
parser.add_argument('--lamb', type=float, default=3.,
                    help='Tradeoff parameters of the plug-and-play algorithm')
parser.add_argument('--sigma_d', type=float, default=(10./255.),
                    help='Denoising level of the DRUNet')
parser.add_argument('--Lx', type=float, default=3/((10/255)**2),
                    help='Penalization term of the linearized ADMM')
parser.add_argument('--eta', type=float, default=0.00001,
                    help='Gradient descent step for the approximate ADMM')
parser.add_argument('--n_iter', type=int, default=100,
                    help='Number of iterations')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='Number of iterations')
parser.add_argument('--n_space', type=int, default=1,
                    help='Number of spatially-varying kernels')
parser.add_argument('--sigma', type=int, default=1.,
                    help='Noise level of the test image')
parser.add_argument('--batch_size', type=int, default=32)

opt = vars(parser.parse_args())



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['gpu_id'])

opt_data = { "phase": "train"
          , "dataroot_H": "/data/datasets_charles/val2014"
          , "dataroot_L": None
          , "H_size": 256
          , "use_flip": True
          , "use_rot": True
          , "scales": [1]
          , "sigma": [opt['sigma'], opt['sigma']+1]
          , "sigma_test": opt['sigma']
          , "n_channels": 3
          , "dataloader_shuffle": True
          , "dataloader_num_workers": 16
          , "dataloader_batch_size": 16
          , "motion_blur": True
          , "coco_annotation_path": "/data/datasets_charles/COCO_annotations/instances_val2014.json"}


# Load data
torch.manual_seed(123456)
random.seed(123456)
np.random.seed(123456)

data_train = data_multiblur(opt_data)
data_test = data_multiblur(opt_data)

data_train.pca_size = 5
data_test.pca_size = 5

train_loader = DataLoader(data_train, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(data_test, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)

# Load model
model = define_Model(opt)

save_folder = os.path.split(opt['save_df_path'])[0]

try:
    os.mkdir(os.path.join(save_folder, opt['model']))
except:
    pass

# Load dataframe
if opt['input_df_path']:
    df = pd.read_csv(opt['input_df_path'], index_col=0)
else:
    df = pd.DataFrame(columns=['psnr_list', 'ssim_list', 'lpips_list' ,'time_list', 'lamb' , 'sigma_d' , 'Lx'])

# Grid search and run model
batch_train = next(iter(train_loader))
model.feed_data(batch_train)

print(60 * '-')
print('Fitting params')
print(60 * '-')

model.fit_params()

psnr = [0 for i in range(model.n_iter)]
ssim = [0 for i in range(model.n_iter)]
lpips = [0 for i in range(model.n_iter)]
time_list = [0 for i in range(model.n_iter)]
c = 0

try: 
    os.mkdir(os.path.join(save_folder, 'LR'))
except:
    pass

try: 
    os.mkdir(os.path.join(save_folder, 'HR'))
except:
    pass

try: 
    os.mkdir(os.path.join(save_folder, 'samples'))
except:
    pass

for batch in tqdm.tqdm(train_loader, desc='Test loop'):
    c += 1
    model.feed_data(batch)
    
    res, res_list, time_batch = model.run()

    
    # Compute metrics
    psnr_batch, ssim_batch, lpips_batch = get_metrics(res_list, batch['H'])
    psnr = list(map(add, psnr, psnr_batch))
    ssim = list(map(add, ssim, ssim_batch))
    lpips = list(map(add, lpips, lpips_batch))
    time_list = list(map(add, time_list, time_batch))

    for i,x in enumerate(res):
        util.imsave(util.tensor2uint(x), os.path.join(save_folder, opt['model'], str(c * 4 + i) + '.png'))
    

    for i in range(len(batch['L'])):
        lr = batch['L'][i]
        hr = batch['H'][i]
        util.imsave(util.tensor2uint(lr), os.path.join(save_folder, 'LR', str(c * 4 + i) + '.png'))
        util.imsave(util.tensor2uint(hr), os.path.join(save_folder, 'HR', str(c * 4 + i) + '.png'))
        with open(os.path.join(save_folder, 'samples', str(c * 4 + i) + '.pickle'), 'wb') as handle:
            pickle.dump({'H': batch['H'][i], 'L': batch['L'][i], 'kmap': batch['kmap'][i],'basis': batch['basis'][i],'sf': batch['sf'][i],'sigma':batch['sigma'][i]},
             handle, protocol=pickle.HIGHEST_PROTOCOL)

    if c == 10:
        break

psnr = list(map(lambda x: x/c, psnr))
ssim = list(map(lambda x: x/c, ssim))
lpips = list(map(lambda x: x/c, lpips))

lamb, sig_d, Lx = model.get_hyperparams()

# Save DataFrame
df.loc[opt['model'], :] = psnr, ssim, lpips, time_list, lamb, sig_d, Lx
df.to_csv(opt['save_df_path'])
