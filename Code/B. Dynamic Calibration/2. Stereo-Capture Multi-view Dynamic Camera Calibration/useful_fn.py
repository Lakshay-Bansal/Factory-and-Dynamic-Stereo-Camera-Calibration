#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:51:36 2023

@author: lakshayb
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_Matchfeature(path, corners, flag, idx):
    image = cv2.imread(path)
    plt.imshow(image)   
    plt.scatter(np.ravel(corners)[::2], np.ravel(corners)[1::2])
    if flag==0:
        plt.title(f"{idx} - Left Image - Detected Corners")
    elif flag==1:
        plt.title(f"{idx} - Right Image - Detected Corners")
    plt.show()


def plot_3D_points(points_3d, i_set):
    fig = plt.figure(figsize=(4,4))
    ax = Axes3D(fig)
    # ax.set_aspect('auto')
    
    x_temp = []
    y_temp = []
    z_temp = list()
    
    for p in points_3d:
        x_temp.append(p[0])
        y_temp.append(p[1])
        z_temp.append(p[2])
            
        
    # ax.scatter(2,3,4) # plot the point (2,3,4) on the figure
    ax.scatter(x_temp, y_temp, z_temp)
    
    # setting title and labels
    ax.set_title(f"{i_set} - Trinagulated 3D points from stereo pair image")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    plt.show()
    
    
def reprojection_3D_pt(path, features, flag, pos, pts_3D, P):
    # Detected corners
    image = cv2.imread(path)
    plt.imshow(image)     
    plt.scatter(np.ravel(features)[::2], np.ravel(features)[1::2])
    if flag ==0:
        plt.title(f"{pos} Left Image - Detected features (Blue) and Reprojected triangulated 3D points (red)")
    elif flag==1:
        plt.title(f"{pos} Right Image - Detected features (Blue) and Reprojected triangulated 3D points (red)")

    
    # Ploting 3D points reprojection
    a=[]
    for idx, pt_3d in enumerate(pts_3D):
        reprojected_pt = np.float32(np.matmul(P, pt_3d))
        reprojected_pt /= reprojected_pt[2]
        a.append(reprojected_pt[0:2])
    
    # plt.imshow(image) 
    for i in range(len(a)):  
        plt.scatter(a[i][0], a[i][1], color='red')
    plt.show()
    
    
def rep_error_fn(P, features, pts_3D):

    error = 0

    for idx, pt_3d in enumerate(pts_3D):
        pt_2d = features[idx]

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]
        
        error += cv2.norm(pt_2d, reprojected_pt[0:2], cv2.NORM_L2)
        # error = error + np.sum((pt_2d - reprojected_pt[0:2])**2)
        
    avg_error = error**0.5 / len(features)    
    return avg_error

#-------------------------------------------------------------------------------#