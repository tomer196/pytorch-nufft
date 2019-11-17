import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.insert(0, '/home/tomerweiss/pytorch-nufft')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import torch
import pytorch_nufft.nufft as nufft
import h5py
import pytorch_nufft.interp as interp

import pytorch_nufft.transforms as transforms

device='cpu'
# create trajectory
x=np.load(f'/home/tomerweiss/PILOT/spiral/4spiral.npy')
x=torch.tensor(x).float()
z=torch.arange(-18,18,36/x.shape[0])
z=z.unsqueeze(1)
x=torch.cat((z,x),dim=1)

# x=np.array(np.meshgrid(np.arange(-18,19)-0.5,np.arange(-160,160),np.arange(-160,160))).T.reshape(-1,3) # trajectory which cover the all kspace
# x=torch.tensor(x).float()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xx=x.numpy()
# ax.plot(xx[:,0],xx[:,1],xx[:,2])
# fig.show()
x=x.to(device)

# Get data
with h5py.File('/home/tomerweiss/Datasets/singlecoil_train/file1001029.h5', 'r') as data:
    # kspace = data['kspace'][:]
    img = data['reconstruction_esc'][:]


img = img.reshape(1,1,37,320,320)
img = torch.tensor(img).to(device)
original_shape=img.shape
plt.figure()
plt.imshow(img[0,0,18,:,:].detach().cpu().numpy(), cmap='gray')
plt.show()

# NUFFT Forward
# ksp = nufft.nufft(img, x, ndim=3, device=device)

kspace=transforms.rfft3(img)
ksp=interp.bilinear_interpolate_torch_gridsample_3d(kspace,x)

# NUFFT Adjoint
img_est = nufft.nufft_adjoint(ksp,x,original_shape, ndim=3,device=device)

plt.figure()
plt.imshow(img_est[0,18,:,:].detach().cpu().numpy(), cmap='gray')
plt.show()
