import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.insert(0, '/home/tomerweiss/pytorch-nufft')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import pytorch_nufft.nufft as nufft
import pytorch_nufft.interp as interp

import pytorch_nufft.transforms as transforms

device='cpu'
# create trajectory
res=256
decimation_rate=4
dt=1e-2
num_measurements=res**2//decimation_rate
x = torch.zeros(num_measurements, 2)
c = decimation_rate / res ** 2 * 1600
r = torch.arange(num_measurements, dtype=torch.float64) * 1e-1
x[:, 0] = c * r * torch.cos(r)
x[:, 1] = c * r * torch.sin(r)
x=x.to(device)

# Get data
img = cv2.imread('DIPSourceHW1.jpg',0).astype('float32')
img = img.reshape(1,1,256,256)
img = torch.tensor(img).to(device)
original_shape=img.shape

# NUFFT Forward
# ksp = nufft.nufft(img, x, device=device)
ksp=transforms.rfft2(img)
ksp=interp.bilinear_interpolate_torch_gridsample(ksp,x)

# NUFFT Adjoint
img_est = nufft.nufft_adjoint(ksp,x,original_shape,device=device)

plt.figure()
plt.imshow(img_est[0,:,:].detach().cpu().numpy(), cmap='gray')
plt.show()
