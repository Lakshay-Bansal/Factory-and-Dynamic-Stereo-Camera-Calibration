#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:15:52 2023

We are going to first find the 2d matched points across all images. 
And then calculate the Rli and Tli of only left images of stereo pair, by E decomposition in each i view.
Use the fix R_s and T_s of stereo setup to define the rotation b/w left and right stereo pair.

Obtain projection matrix for each image of each view. Then using all 2D match of one same feature in all 
left and right image and along different view compute the 3D world point of that feature.
"""
import os
import time

# PWD = os.getcwd()
# sys.path.append(rf'{PWD}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from glob import glob
from time import time
from pprint import pprint
import sys
from random import sample
import shutil

from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
from bundle_adjustment import bundle_adjustment, get_intrinsics
from utils import plot_CheckerBoard_corners, get_camera_intrinsic_params, rep_error_fn, calibrate_fisheye, tprint, reprojection_3D_pt, fit_3D_plane
from triangulation_nView import triangulation_nView
from threshold_view_selection import threshold_view_selection


## Find Checkerboard Corners

camera_ip = ["192.168.1.72", "192.168.1.69"]

imagesL = glob(rf"D:\Lakshay\M.Tech\M.Tech - PGP\Dataset\1_Dataset_Vehant_StereoCamera\Checkerboard_Pattern\Dataset_1\image_*_{camera_ip[0]}.jpg")
imagesR = glob(rf"D:\Lakshay\M.Tech\M.Tech - PGP\Dataset\1_Dataset_Vehant_StereoCamera\Checkerboard_Pattern\Dataset_1\image_*_{camera_ip[1]}.jpg")
CHECKERBOARD = (14,9)
CORNERS = CHECKERBOARD[0] * CHECKERBOARD[1]
squareSize = 7
DIM = (1920, 1200)

# imagesL_name = os.listdir("/mnt/Data2/Datatset/ZhangData/left/")
# imagesL = [os.path.join("/mnt/Data2/Datatset/ZhangData/left/", imgName) for imgName in imagesL_name]

# imagesR_name = os.listdir("/mnt/Data2/Datatset/ZhangData/right/")
# imagesR = [os.path.join("/mnt/Data2/Datatset/ZhangData/right/", imgName) for imgName in imagesR_name]

# CHECKERBOARD = (9,6)
# CORNERS = CHECKERBOARD[0] * CHECKERBOARD[1]
# squareSize = 5
# DIM = (640,480)

imagesL.sort()
imagesR.sort()
 
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)


objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp*squareSize

objpoints = []
imgpointsL = []
imgpointsR = []

# To store the path of only those images whose corner are detected successfully
# Then at the end will update the imagesL and imagesR with new paths
imagesL_new = []
imagesR_new = []

for i in range(len(imagesL)):
    try:
        imgL = cv2.imread(imagesL[i])
        imgR = cv2.imread(imagesR[i])
        
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        
        flags_corners =  (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.fisheye.CALIB_CHECK_COND
        )
        
        retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, flags_corners)
        retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, flags_corners)
    
        if (retL and retR):
            objpoints.append(objp)
            cv2.cornerSubPix(grayL, cornersL, (3,3), (-1,-1), subpix_criteria)
            cv2.cornerSubPix(grayR, cornersR, (3,3), (-1,-1), subpix_criteria)
            
            cornersL = cornersL.reshape(1,CORNERS,2)
            cornersR = cornersR.reshape(1,CORNERS,2)

            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            
            # To plot the detected corners plot_CheckerBoard_corners(path, corners, flag, idx)
            # Left
            plot_CheckerBoard_corners(imagesL[i], imgpointsL[-1], 0, i)
            imagesL_new.append(imagesL[i])
            # Right
            plot_CheckerBoard_corners(imagesR[i], imgpointsR[-1], 1, i)
            imagesR_new.append(imagesR[i])
            
            sys.stdout.write(f"\r{i+1} : {cornersL.shape}")
            sys.stdout.flush()
    except:
        print(sys.exc_info())
        continue

N_OK = len(imgpointsL)

imagesL = imagesL_new
imagesR = imagesR_new


# Calibrate both cameras

FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_CHECK_COND
    + cv2.fisheye.CALIB_FIX_SKEW
    + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
)

#  Calibrating left and right camera individually
k = np.array([[700.0,0.0,800.0],
              [0.0,700.0,500.0],
              [0.0,0.0,1.0]])
d = np.array([[0.02],
              [0.02],
              [0.02],
              [0.02]])


# last _, _ will hold the R and T between left images as a stereo pair
rmsL, K_L, D_L, rvecs_L, tvecs_L = calibrate_fisheye(imgpointsL, objpoints, DIM, k, d, FLAGS)


rmsR, K_R, D_R, rvecs_R, tvecs_R = calibrate_fisheye(imgpointsR, objpoints, DIM, k, d, FLAGS)

# Find the R and T of stereo Setup

# Use cv2.fisheye.stereoCalibrate for images of calibration pattern that are more barel distorted 
# while cv2.stereoCalibrate for more of a undistorted checkerborad pattern


parameters = \
        cv2.stereoCalibrate(
            objpoints,
            imgpointsL, imgpointsR,
            K_L, D_L,
            K_R, D_R,
            DIM,
            flags=cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )

# parameters = \
#         cv2.fisheye.stereoCalibrate(
#             objpoints,
#             imgpointsL, imgpointsR,
#             K_L, D_L,
#             K_R, D_R,
#             DIM,
#             flags=cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_USE_INTRINSIC_GUESS,
#             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e0),
#         )
     
# Change 1e-6 to 1e1.
# The cv::TermCriteria object specifies the termination criteria for the 
# iterative optimization algorithm used by stereoCalibrate. 
# It has three components: the maximum number of iterations, 
# the maximum change in the parameters, and the minimum error tolerance. 
# The abs_max < threshold assertion failure in the error message you received 
# indicates that the maximum change in the parameters has exceeded the specified threshold.

print("\n\nIntrinsic matrix: ")
K = get_camera_intrinsic_params(parameters)
print("###K####\n\n", K)


print("\n\nStereo setup extrinsic parameters: ")
R_s = parameters[5]
print("###R####\n\n", R_s)

T_s = parameters[6]
print("\n###T####\n\n", T_s)

# Output of above command if don't want to do calibration again and again for the stereo setup
# Stereo setup extrinsic parameters ofr Dataset 1: 
# ###R_s####

#  [[ 9.98689287e-01 -1.11817931e-03 -5.11708584e-02]
#  [ 3.59358324e-04  9.99889876e-01 -1.48359611e-02]
#  [ 5.11818125e-02  1.47981268e-02  9.98579710e-01]]

# ###T_s####

#  [[ 4.82310848]
#  [-0.05925189]
#  [-0.01181658]]



"""_________________________________________________________"""
# SFM on Left Images of stereo pairs

# Now we will calculate the R_l and T_l amog the left images of the stereo pair image
# Then compute the R_r and T_r by using R_s and T_s of the stereo pair as calculated above 
# by inbuilt stereo calibartion

# K = np.array(get_camera_intrinsic_params(parameters), dtype=np.float)
R_l_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
# R_l stores the rotation matrix between left images of stereo images captured in a sfm
R_l = []
R_l.append(R_l_0)


P2 = np.zeros((3,4))

# For Left camera of stereo pair
projectionMatrices_L_before = []
# projectionMatrices_after = []

# For Right camera of stereo pair
projectionMatrices_R_before = []
# projectionMatrices_R_after = []

RPE = []

# As there are 25 images I am running the code on 8 to 10 images
interval = 1
left_img_idx = list(range(0, N_OK, interval))

for view in left_img_idx[1:]:
    print(int(view/interval))
    
    # See we are taking only left images to calculate the E
    # Computing R_l and T_l of sfm to left image with another left image 
    # R_l_1 and by pre-multiplying R_l_1 with [R_s T_s] gives the rotation matrix of right image
    pts1 = np.array(imgpointsL[0][0][:])    # We estimating R_l and T_l wrt to left image of view 1 in every view
    pts2 = np.array(imgpointsL[view][0][:])
    
    # image = cv2.imread(imagesL[left_img_idx[view+1]])
    # plt.imshow(image)   
    # plt.scatter(np.ravel(imgpointsL[left_img_idx[view+1]][0][:])[::2], np.ravel(imgpointsL[left_img_idx[view+1]][0][:])[1::2])
    # plt.title(f"{left_img_idx[view]} Left Image - Detected Corners")
    # plt.show()
    
    plot_CheckerBoard_corners(imagesL[view], imgpointsL[view], 0, view)
    
    
    # F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    # print("The fundamental matrix \n" + str(F))
    
    # E = np.matmul(np.matmul(K.T, F), K)
    # print("The essential matrix is \n" + str(E))
    
    # Directly computing E matrix
    E, mask = cv2.findEssentialMat(pts1,
                              pts2, 
                              K,
                              method=cv2.RANSAC, 
                              prob=0.99,
                              threshold=0.75)
    
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    retval, R_l_temp, t_l_temp, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    # R_l_1 = np.zeros((3,4))
    # R_l_1[:3,:3] = R_l_temp
    # R_l_1[:3, 3] = t_l_temp.ravel()
    # R_l.append(R_l_1)
    # print(f"{view+1}:     The R_l_1 \n" + str(R_l_1))
    
    # Create 3*4 matrix for Left image defined by E
    # Correct P_l = K [R_l -R_l C], where C is the camera centre defined as C = -R_L^-1 t_L
    # Hence correct P_l = K [R_l -R_l -R_L^-1 t_L] = K [R_l t_l]
    R_l_1 = np.hstack([R_l_temp, t_l_temp])
    R_l.append(R_l_1)

    # P2 = np.matmul(K, R_l_1)
    # Just passing [R_l T_l] as projection matrix for BA improvement
    # While K will be passed as fixed shape that doesn't update in the BA
    # Will used in projecting the 3D points on the image plane
    P2 = R_l_1

    # print("The projection matrix 1 \n" + str(P1))
    print(f"\n\n{int(view/interval)}: The projection matrix left \n" + str(P2))
    
    projectionMatrices_L_before.append(P2)
    
    # R_r is obatined by the R_s and T_s of the camera setup on the R_l
    # One way
    # R_r_temp = np.zeros((3,4))verbose
    # R_r_temp[:3, :3] = np.dot(R, R_l_temp)
    # R_r_temp[:3, 3] = t_l_temp.ravel() + T.ravel()
    
    # Correct P_r = K [R -R C]|R_l -R_l C_L|from scipy.linalg import svd
   
    #                         | 0     1    |, where 
    # C is the camera centre defined as C = -R^-1 T and C_L = -R_L^-1 t_L
    # Hence correct P_l = K [R_s T_s] |R_l t_L|
    #                             | 0    1|
    # So the implementation in main_by K E_v2.py is correct implementation
    
    # R_r is obatined by the R_s and T_s of the camera setup on the R_l_1
    R_r_temp = np.hstack([R_s, T_s])
    R_r_temp2 = np.vstack([R_l_1, [0, 0, 0, 1]])
    
    R_r = np.matmul(R_r_temp, R_r_temp2)

    # projectionMatrices_R_before.append(np.matmul(K, R_r_temp))
    projectionMatrices_R_before.append(R_r)

    # Ploting the corners on the image
    pts1 = np.transpose(pts1)
    pts2 = np.transpose(pts2)

    print("Shape pts 1\n" + str(pts1.shape))
    
    

# Common 3D world coordinates for matched 126 points in images from many views
pts_2d_left = []
pts_2d_right = []

# As there are 25 images I am running the code on 8 to 10 images
# left_img_idx = list(range(0, N_OK, 3))

# Not taking 0 index images match points as that is Refrence with R = I and T = 0
for view in left_img_idx[1:]:
    # print(view+1)
    pts_2d_left.append(np.array(imgpointsL[view][0][:]))
    pts_2d_right.append(np.array(imgpointsR[view][0][:]))



# Stereo Camera calibration - MultiView Images

"""_________________________________________________________"""
# Triangulation from n Views

common_3D_pts_svd = triangulation_nView(pts_2d_left, pts_2d_right, projectionMatrices_L_before, projectionMatrices_R_before, K)

common_3D_pts = common_3D_pts_svd

# Projecting 3D points to image plane
for view in range(len(left_img_idx)-1):
    
    P_l = projectionMatrices_L_before[view]
    P_r = projectionMatrices_R_before[view]

    # left_image = cv2.imread(imagesL[left_img_idx[view+1]])
    # plt.imshow(left_image)     
    # # plt.scatter(np.ravel(imgpointsL[left_img_idx[view+1]][0][:])[::2], np.ravel(imgpointsL[left_img_idx[view+1]][0][:])[1::2])
    # plt.scatter(np.ravel(pts_2d_left[view][:])[::2], np.ravel(pts_2d_left[view][:])[1::2])
    # plt.title(f"{left_img_idx[view+1]} Left Image - Detected Corners")
    # plt.show()
    
    # a=[]
    # for idx, pt_3d in enumerate(common_3D_pts):
    #     reprojected_pt = np.float32(np.matmul(P_l, pt_3d))
    #     reprojected_pt /= reprojected_pt[2]
    #     a.append(reprojected_pt[0:2])
    
    # plt.imshow(left_image) 
    # for i in range(len(a)):  
    #     plt.scatter(a[i][0], a[i][1])
    # plt.title(f"{left_img_idx[view+1]} Left Image - Reprojected common 3D points in all views")
    # plt.show()
    
    # Above comment steps are performed in the below function
    # reprojection_3D_pt(path, corners, flag, idx, pts_3D, K, P)
    idx = left_img_idx[view+1]
    reprojection_3D_pt(path = imagesL[idx], corners = pts_2d_left[view][:], flag=0, pos=idx, 
                       pts_3D = common_3D_pts, K = K, P = P_l)

    # Similarily for right image
    reprojection_3D_pt(path = imagesR[idx], corners = pts_2d_right[view][:], flag=1, pos=idx, 
                       pts_3D = common_3D_pts, K = K, P = P_r)


# Thresholding for view selection

final_view = threshold_view_selection(projectionMatrices_L_before, projectionMatrices_R_before, pts_2d_left, pts_2d_right,
                             imagesL, imagesR, common_3D_pts, K, left_img_idx, 3800)


"""_________________________________________________________"""
# Stereo bundle adjustment
P_BA = [] 

# Appending the camera intrinsic matrix as a unkonwn for stereo BA
# P_BA.extend(np.array(K).ravel())
# Just focal length and camera offset will get updated during BA
P_BA.extend(np.array([ K[0, 0], K[0, 2], K[1, 2] ]))

# Now pasiing the R_s and T_s of stereo camera setup for BA
# So that it will give the R and T of setereo setup directly
# Rather using decomposing P after BA and further decomposing to obatin R and T
# Then we will have R and T for every used stereo pair
P_BA.extend(np.array(cv2.Rodrigues(R_s)[0]).ravel())    # Just storing angles from R_s matrix
C_s = -1*np.matmul(R_s.T, T_s)
P_BA.extend(np.array(C_s).ravel())  # We are storing camera centre as unknown in BA

R_L = cv2.Rodrigues(np.eye(3))[0]   # Identity matrix
P_BA.extend(np.array(R_L).ravel())

# for idx in range(len(projectionMatrices_L_before)):
#     P_l = projectionMatrices_L_before[idx].ravel()
#     P_BA.extend(P_l)

# for idx in range(len(projectionMatrices_R_before)):
#     P_r = projectionMatrices_R_before[idx].ravel()
#     P_BA.extend(P_r)

# We just use the views that have low reprojection error
for fn_view in final_view:
    
    P_l = projectionMatrices_L_before[round(fn_view / interval)-1]
    # 1. Rather then using complete Rotation matrix we will use its Rodriques angles only - 3 Parameters
    
    # Left view rotation - Rodrigues angles
    r_L_angle, _ =	cv2.Rodrigues( P_l[:3, :3] ) 
    P_BA.extend(r_L_angle.ravel())
    # Left view - translation
    C_l = -1*np.matmul(P_l[:3, :3].T, P_l[:3, -1]) 
    # Instead of camera translation storing camera centre of left view images in SFM
    P_BA.extend(C_l.ravel())


# for P_r in projectionMatrices_R_before:

#     # 1. Rather then using complete Rotation matrix we will use its Rodriques angles only - 3 Parameters    
#     r_R_angle, _ =	cv2.Rodrigues( P_r[:3, :3] ) 
#     P_BA.extend(r_R_angle.ravel())
#     # Left view - translation
#     P_BA.extend(P_r[:3, -1].ravel())


# BA n View
# Now to perform BA along all views and taking P of left and Right camera
# we need to pass the list conatining all the P matrix and then 
# Need to calculate the residual for all the P and 3D points
# 3D points will be used in cartesian coordinate form then in homogeneous coordinates   
import time
 
t0 = time.time()

P_after_BA, common_3D_pts_after_BA, corrected_values = bundle_adjustment(
    np.array(common_3D_pts_svd[:, :3]), 
    np.array(pts_2d_left)[list((np.array(final_view)/interval).astype('int') -1)], 
    np.array(pts_2d_right)[list((np.array(final_view)/interval).astype('int') -1)], 
    P_BA, K)    

t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))

#Fitting a plane to the obatined 3D points after BA
fit_3D_plane(common_3D_pts_after_BA)


"""_________________________________________________________"""
# Now ploting the corrected 3D points after BA and reprojected points from them in image plane
# Seperating the P_l_afterBA and P_r_afterBA

K_after_BA = get_intrinsics( P_after_BA[0:3] )

R_after_BA = cv2.Rodrigues( P_after_BA[3:6] )[0]
C_after_BA = P_after_BA[6:9].reshape(3,1)
T_after_BA = -1*np.matmul(R_after_BA, C_after_BA)

R_L_after_BA = cv2.Rodrigues( P_after_BA[9:12] )[0]

nViews= int( (len(P_after_BA) - 12) / 6 ) # -12 because 12 parameter of stereo setup K, C, R_s and T_s

P_l_afterBA = P_after_BA[12:].reshape(nViews,2,3)   # As at 0 now have [R_s T_s] the setereo setup 


temp = []

for p in common_3D_pts_after_BA:
    temp.append(  np.hstack([p, 1]) )


common_3D_pts_after_BA = temp


for idx, fn_view in enumerate(final_view):
    print(f"{fn_view} - View valid for BA ")
    r_L_matrix , _ = cv2.Rodrigues( P_l_afterBA[idx][0] )
    C_L = P_l_afterBA[idx][1].reshape((3, 1))
    P = np.hstack([r_L_matrix, -1*np.matmul(r_L_matrix, C_L)])
    
    dummy = np.hstack([R_L_after_BA, np.zeros((3,1)) ])
    dummy2 = np.vstack([P, [0, 0, 0, 1]])
    P_l = np.matmul(dummy, dummy2)
    
    # r_R_matrix , _ = cv2.Rodrigues( P_r_afterBA[view][0] )
    # P_r = np.hstack([r_R_matrix, P_r_afterBA[view][1].reshape((3,1)) ])
    
    dummy = np.hstack([R_after_BA, T_after_BA])
    # dummy2 = np.vstack([P, [0, 0, 0, 1]])
    P_r = np.matmul(dummy, dummy2)
    
    # Above comment steps are performed in the below function
    # reprojection_3D_pt(path, corners, flag, idx, 3D_pts, K, P)
    # reprojection_3D_pt(path = imagesL[fn_view], corners = pts_2d_left[fn_view-1][:], flag=0, pos=fn_view, 
    #                    pts_3D = common_3D_pts_after_BA, K = K_after_BA, P = P_l)

    # # Similarily for right image
    # reprojection_3D_pt(path = imagesR[fn_view], corners = pts_2d_right[fn_view-1][:], flag=1, pos=fn_view, 
    #                    pts_3D = common_3D_pts_after_BA, K = K_after_BA, P = P_r)
    
    # If involved interval
    i = round(fn_view / interval)
    # print(i)
    reprojection_3D_pt(path = imagesL[fn_view], corners = pts_2d_left[i-1][:], flag=0, pos=fn_view, 
                       pts_3D = common_3D_pts_after_BA, K = K_after_BA, P = P_l)

    # Similarily for right image
    reprojection_3D_pt(path = imagesR[fn_view], corners = pts_2d_right[i-1][:], flag=1, pos=fn_view, 
                       pts_3D = common_3D_pts_after_BA, K = K_after_BA, P = P_r)


#####-------------------------------------------------------------------------------------------------------------------_####
# Error Analysis


# Reprojection Error for final views
rpe = {'beforeBA': [], 'afterBA': []}

# RPE Before BA

# for view in final_view:
for view in np.round(np.array(final_view) / interval).astype(int):    
    # print(view)
    P_l = projectionMatrices_L_before[view-1]
    P_r = projectionMatrices_R_before[view-1]
    
    P_LR = [P_l, P_r]
    points_2d_LR = [pts_2d_left[view-1], pts_2d_right[view-1]]
    
    for P, points_2d in zip(P_LR, points_2d_LR):
        # print(points_2d.shape)
        error = 0
        for idx, pt_3d in enumerate(common_3D_pts_svd):
            # print(pt_3d.shape)
            # Kth view, xth camera and jth matched feature 
            pt_2d = np.array([points_2d[idx][0], points_2d[idx][1]])
            reprojected_pt = np.matmul(np.matmul(K,P), pt_3d)
            reprojected_pt /= reprojected_pt[2]
            # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
            error = error + np.sum((pt_2d - reprojected_pt[0:2])**2)
        
        avg_error = error**0.5 / CORNERS
        rpe['beforeBA'].append(avg_error)


# RPE after BA

for pos, fn_view in enumerate(final_view):
    
    # P_l = P_l_afterBA[view]
    # P_r = P_r_afterBA[view]
    
    r_L_matrix , _ = cv2.Rodrigues( P_l_afterBA[pos][0] )
    C_L = P_l_afterBA[pos][1].reshape((3, 1))
    P = np.hstack([r_L_matrix, -1*np.matmul(r_L_matrix, C_L)])
    P_l = P
    
    # dummy = np.hstack([R_L_after_BA, np.zeros((3,1)) ])
    dummy = np.hstack([R_after_BA, T_after_BA])
    dummy2 = np.vstack([P, [0, 0, 0, 1]])
    
    P_r = np.matmul(dummy, dummy2)
    
    P_LR = [P_l, P_r]
    points_2d_LR = [pts_2d_left[round(fn_view / interval)-1], pts_2d_right[round(fn_view / interval)-1]]
    
    for P, points_2d in zip(P_LR, points_2d_LR):
        # print(points_2d.shape)
        error = 0
        for idx, pt_3d in enumerate(common_3D_pts_after_BA):
            # print(pt_3d.shape)
            # Kth view, xth camera and jth matched feature 
            pt_2d = np.array([points_2d[idx][0], points_2d[idx][1]])
            reprojected_pt = np.matmul(np.matmul(K_after_BA,P), pt_3d)
            reprojected_pt /= reprojected_pt[2]
            # print(reprojected_pt)
            # print("Reprojection Error \n" + str(pt_2d - repr# # Normalising P
            error = error + np.sum((pt_2d - reprojected_pt[0:2])**2)
        
        avg_error = error**0.5 / CORNERS
        # print(avg_error)
        rpe['afterBA'].append(avg_error)

# Average RPE

sum_rpe = np.sum(np.array(rpe['beforeBA']))
length = len(rpe['beforeBA'])
print("Average re-projection error before BA: ", sum_rpe / length)

sum_rpe = np.sum(np.array(rpe['afterBA']))
length = len(rpe['afterBA'])
print("Average re-projection error after BA: ", sum_rpe / length)
