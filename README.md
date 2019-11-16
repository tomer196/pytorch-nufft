# pytorch-nufft: PyTorch implemenation of Non-uniform Fast Fourier Transform (Nu-FFT)

This repository contains a PyTorch implementation of Nu-FFT, currently only for 2D.
Our implementation his deferential according to the coordinates of the measurements, meaning, when using the 
transform in GD based optimization methods you can update the coordinates of the measurements according to the loss optimization. 
We used [Sigpy](https://github.com/mikgroup/sigpy) as base for our implementation.

## Usage

To understant the use of pytorch-nufft we addvise to run the example script.
```bash
python nufft-test.py
```
