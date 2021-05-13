import numpy as np
import torch
import cv2
from visionPipeline.main import Vision_Control

vision_module = Vision_Control()

mod = Vision_Control()
img_dir = "a_2.jpg"
_og_img  = cv2.imread(img_dir)
# mod.showOutput = 1
for i in range(10):
    mod.getTrackPoints(_og_img)
    print(mod.time_taken)