import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class f2t_VAE(nn.Module):

    def __init__(self, features,bs,H1,H2,W, device = torch.device('cuda')):
        super(f2t_VAE, self).__init__()
        self.device = device
        self.nz = features
        self.bs = bs
        self.H1 = H1
        self.H2 = H2
        self.W = W
        encoderModules = []
        self.hiddenDims = [4, 8, 16, 32]
        hiddenDims_bk = [32,16,8,4]
        inChannels = 1
        for hDim in self.hiddenDims:
            encoderModules.append(
                nn.Sequential(
                    nn.Conv2d(inChannels, hDim, kernel_size= 1, stride= 2),
                    nn.BatchNorm2d(hDim),
                    nn.LeakyReLU()
                )
            )
            inChannels = hDim
        self.encoder = nn.Sequential(*encoderModules)


        # 100 x 216

        # self.hiddenDims.reverse()
        decoderModules = []
        for i in range(len(hiddenDims_bk)-1):
            decoderModules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddenDims_bk[i], hiddenDims_bk[i + 1], kernel_size=3, stride = 2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hiddenDims_bk[i + 1]),
                    nn.LeakyReLU())
            )

        decoderModules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddenDims_bk[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*decoderModules)
    

        self.fc1 = nn.Linear(self.hiddenDims[-1]*self.H1//(2**len(self.hiddenDims))*self.W//(2**len(self.hiddenDims)), 64)
        self.fc21 = nn.Linear(64, features)
        self.fc22 = nn.Linear(64, features)

        self.fc3 = nn.Linear(features, 64)
        self.fc4 = nn.Linear(64, self.hiddenDims[-1]*self.H2//(2**len(self.hiddenDims))*self.W//(2**len(self.hiddenDims)))

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # print("ENCODER------------")
        # print(x.shape)
        # print(x.shape)
        conv = self.encoder(x)
        # print(conv.shape)
        conv = conv.view(self.bs,-1)
        # print("After encoder = ", conv.size()) #torch.Size([1, 256, 2, 2])
        # print(conv.shape)
        h1 = self.fc1(conv)
        # print("FC1 after encoder = ", h1.size()) #torch.Size([1, 512])
        means = self.fc21(h1)
        vars = self.fc22(h1)
        # print("Means = ", means.size()) #torch.Size([1, 10])
        # print("Vars = ", vars.size()) #torch.Size([1, 10])
        
        return means, vars


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        ret = mu + eps*std
        return ret


    def decode(self, z):
        # print("DECODER---------")
        # print(self.hiddenDims[-1], self.H2//(2**len(self.hiddenDims)), self.W//(2**len(self.hiddenDims)))
        # print("z", z.size()) #torch.Size([batch, nz])
        h3 = self.relu(self.fc3(z))
        # print("h3", h3.size()) #torch.Size([batch, 512])
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(self.bs, 32, 10, 20)
        # print(deconv_input.shape)

        # print("deconv_input", deconv_input.size()) #torch.Size([batch, 1024])
        decoded = self.decoder(deconv_input)
        # print("decoded", decoded.size()) #torch.Size([1440, 3, 28, 28]
        # print(decoded.shape)
        return decoded



    def encodeAndReparametrize(self, x):
        # print("INPUT = ", x.size()) 
        mu, logvar = self.encode(x) #torch.Size([batch, nz])
        z = self.reparametrize(mu, logvar) #torch.Size([batch, nz])
        # print('REPARAM SIZE = ', z.size())
        return z, mu, logvar

    def forward(self, x):
        # print(x.shape)
        z, mu, logvar = self.encodeAndReparametrize(x)
        # print(mu.shape)
        decoded = self.decode(z)
        return decoded, mu, logvar, z


# if __name__ == '__main__':
#     vae = CNNVAE(5).cpu()
#     x = torch.randn((1, 3, 128, 96))
#     vae(x)