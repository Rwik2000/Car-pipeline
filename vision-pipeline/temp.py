import numpy as np
import torch
import cv2
from utils import find_bezier_trajectory
import time

traj_model = torch.load("./models/trajec_model_100.pt")
if torch.cuda.is_available():
    traj_model.cuda()
enc_model = torch.load("./models/vae_model_30.pt")
enc_model.bs = 1

def findTrajectory(_og_img):
    og_img = _og_img.copy()
    _og_img = cv2.resize(_og_img,(200, 152))
    _og_img = _og_img/255
    _og_img = _og_img.transpose(2,0,1)
    _og_img = torch.Tensor(_og_img).cuda().unsqueeze(0)
    _mu,_logvar = enc_model.encoder(_og_img)
    enc = enc_model.sampling(_mu,_logvar)
    inp = enc
    out = traj_model(inp)
    out = out.detach().cpu().numpy()
    out= out.reshape(2,-1).T
    out = (out*[800,600]).astype(int)
    out = find_bezier_trajectory(out, 20)
    for i in range(len(out)-1):
        cv2.line(og_img,tuple(out[i]), tuple(out[i+1]), (255,0,0), 3)
    return out

img_dir = "VAE_img_139.jpg"
_og_img  = cv2.imread(img_dir)

for i in range(10):
    st_time = time.time()
    findTrajectory(_og_img)
    print(time.time()-st_time)

