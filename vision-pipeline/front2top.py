import numpy as np
from numpy.lib.twodim_base import tril_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from neuralNet import CNNVAE
from torchvision import datasets, transforms
import cv2
import os
from tqdm import tqdm


BATCH_SIZE = 1
LATENT_SIZE = 20
H1 = 160
H2 = 160
W = 320

# build model
vae = CNNVAE(LATENT_SIZE, BATCH_SIZE, H1,H2, W)
vae = torch.load("models/F2T_90.pt")

if torch.cuda.is_available():
    vae.cuda()


inp_img  = cv2.imread("a_2.jpg")
inp_img = cv2.resize(inp_img,(W,H1))
inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('inp', inp_img)

inp_img = inp_img/255

inp_img = torch.Tensor([[inp_img]]).to('cuda')

vae.bs = 1
out,_,_,_ = vae.forward(inp_img)
out = out.detach().cpu().numpy()
out = out[0][0]*255
_,out = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY)
out = cv2.resize(out, (800,600))


cv2.imshow('a',out)
cv2.waitKey(0)
# cv2.imwrite("inference/B_"+str(epoch)+".jpg", out)
    # vae.bs = BATCH_SIZE