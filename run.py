import numpy as np
import torch
import cv2
from visionPipeline.vision_module import Vision_Control

vision_module = Vision_Control()
img_dir = "a_2.jpg"
_og_img  = cv2.imread(img_dir)

for i in range(10):
    vision_module.getTrackPoints(_og_img)
    print(vision_module.time_taken)