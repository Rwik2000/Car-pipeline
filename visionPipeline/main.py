import numpy as np
import cv2
import torch
import time

from .utils import find_bezier_trajectory

def processCamImage():
    pass
class Vision_Control():
    def __init__(self,):
        self.traj_model = torch.load("./models/trajec_model_80.pt")
        self.enc_model = torch.load("./models/vae_model_90.pt")
        self.f2t_model = torch.load("./models/F2T_100.pt")
        self.enc_model.bs = 1
        self.f2t_model.bs = 1

        if torch.cuda.is_available():
            self.traj_model.cuda()
            self.enc_model.cuda()
            self.f2t_model.cuda()

        self.numBezPts = 20 #number of bezier points in the output
        self.time_taken = 0

        self.showOutput = 1
        self.waitkey = 0
        

    def findTrajectory(self, image):
        # Preparing the image 
        if self.showOutput:
            k = image.copy()
        image = image/255
        image = image.transpose(2,0,1)
        image = torch.Tensor(image).cuda().unsqueeze(0)

        # Encoding the top view
        _mu,_logvar = self.enc_model.encoder(image)
        enc = self.enc_model.sampling(_mu,_logvar)

        # Converting the encodings to trajectories
        crude_points = self.traj_model(enc)

        # Resizing the preparing the output
        crude_points = crude_points.detach().cpu().numpy()
        crude_points = crude_points.reshape(2,-1).T
        crude_points = (crude_points*[800,600]).astype(int)

        # Smoothening the output with Bezier
        trk_points = find_bezier_trajectory(crude_points, self.numBezPts)

        # Adding the points to the image and showing the path if the showOutput toggle in ON.
        # For real-time detection no Need of keeping it ON.
        if self.showOutput:
            k = cv2.resize(k, (800,600))
            for i in range(len(trk_points)-1):
                k = cv2.line(k,tuple(trk_points[i]), tuple(trk_points[i+1]), (255,0,0), 3)
            cv2.imshow('final', k)
            cv2.waitKey(self.waitkey)

        return trk_points
    
    def cvt_F2T(self,inp_img):
        # preparing the image for the front to top model.
        H = 160
        W = 320
        inp_img = cv2.resize(inp_img,(W,H))
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
        inp_img = inp_img/255
        inp_img = torch.Tensor([[inp_img]]).to('cuda')

        # Sending the processed input through the front 2 top model.
        out,_,_,_ = self.f2t_model.forward(inp_img)

        # processing the image for further models
        out = out.detach().cpu().numpy()
        out = out[0][0]*255
        _,out = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY)
        # NOTE: Image size is kept such that it is compatible for the trajectory model.
        out = cv2.resize(out, (200,152))
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return out
    
    def getTrackPoints(self, lane_img):
        st_time = time.time()
        top_img = self.cvt_F2T(lane_img)
        out = self.findTrajectory(top_img)
        self.time_taken = time.time() - st_time
        return out

# mod = Vision_Control()
# img_dir = "img_196.jpg"
# _og_img  = cv2.imread(img_dir)
# # mod.showOutput = 1
# for i in range(10):
#     mod.getTrackPoints(_og_img)
#     print(mod.time_taken)
