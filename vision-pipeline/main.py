import numpy as np
import cv2
import torch
import time

from utils import find_bezier_trajectory

traj_model = torch.load("./models/trajec_model_80.pt")
enc_model = torch.load("./models/vae_model_90.pt")
f2t_model = torch.load("./models/F2T_90.pt")

enc_model.bs = 1
f2t_model.bs = 1

if torch.cuda.is_available():
    traj_model.cuda()
    enc_model.cuda()
    f2t_model.cuda()

def findTrajectory(_og_img):
    # og_img = _og_img.copy()
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
    return out

def cvt_F2T(inp_img):
    H1 = 160
    W = 320
    inp_img = cv2.resize(inp_img,(W,H1))
    inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    inp_img = inp_img/255

    inp_img = torch.Tensor([[inp_img]]).to('cuda')
    out,_,_,_ = f2t_model.forward(inp_img)
    out = out.detach().cpu().numpy()
    out = out[0][0]*255
    _,out = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY)
    out = cv2.resize(out, (800,600))
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    return out



img_dir = "a_2.jpg"
_og_img  = cv2.imread(img_dir)

for i in range(10):
    st_time = time.time()
    top_img = cvt_F2T(_og_img)
    out = findTrajectory(top_img)
    print(time.time()-st_time)
