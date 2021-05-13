import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class topview_VAE(nn.Module):
    def __init__(self, H, W, LS, BS):
        super(topview_VAE, self).__init__()
        self.H = H
        self.W = W
        self.ls = LS
        self.bs = BS
        # encoder part
        self.cv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cv2 = nn.Conv2d(16, 16, 3, 2, 1)
        self.cv3 = nn.Conv2d(16, 16, 3, 2, 1)

        self.fc1 = nn.Linear(H//8*W//8*16, 32)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(32, self.ls)
        self.fc32 = nn.Linear(32, self.ls)
        # decoder part
        self.fc4 = nn.Linear(self.ls, 32)
        # self.fc5 = nn.Linear(64, 128)
        # self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(32, H//8*W//8*16)
        self.cvL1 = nn.ConvTranspose2d( 16, 16, 3, 2, 1, 1)
        self.cvL2 = nn.ConvTranspose2d( 16, 16, 3, 2, 1, 1)
        self.cvL3 = nn.ConvTranspose2d( 16, 3, 3, 2, 1, 1)

        
    def encoder(self, x):
        h = F.relu(self.cv1(x))
        h = F.relu(self.cv2(h))
        h = F.relu(self.cv3(h))

        h = h.view(self.bs, -1)
        h = F.relu(self.fc1(h))
        # h = F.relu(self.fc2(h))
        # h = F.relu(self.fc3(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        # h = F.relu(self.fc5(h))
        # h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = h.view(self.bs, 16, self.H//8,self.W//8)
        h = F.leaky_relu(self.cvL1(h))
        h = F.leaky_relu(self.cvL2(h))        
        h = torch.sigmoid(self.cvL3(h))
        return h
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var