import cv2
import numpy as np
from scipy.linalg import pascal
import pickle
import os

def find_bezier_trajectory(coordinates, points):
    n=len(coordinates)

    pascal_coord=pascal(n,kind='lower')[-1]
    t=np.linspace(0,1,points)

    bezier_x=np.zeros(points)
    bezier_y=np.zeros(points)

    for i in range(n):
        k=(t**(n-1-i))
        l=(1-t)**i
        bezier_x+=np.multiply(l,k)*pascal_coord[i]*coordinates[n-1-i][0]
        bezier_y+=np.multiply(l,k)*pascal_coord[i]*coordinates[n-1-i][1]
    bezier_xd=[]
    bezier_yd=[]
    for i in range(len(bezier_x)):
        bezier_xd.append(int(bezier_x[i]))
        bezier_yd.append(int(bezier_y[i]))

    bezier_coordinates = np.transpose([bezier_xd, bezier_yd])
    # print(bezier_coordinates)
    return bezier_coordinates