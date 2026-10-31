#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Sat Apr 27 2023

# @author: laksh
# """

# This code is modified in the following aspects for treating unknown during bundle adjustment:
    # 1. Rather then using complete Rotation matrix we will use its Rodriques angles only - 3 Parameters
    # 2. 3D points will be used in cartesian coordinate form then in homogeneous coordinates - 3 Parameter
    # 3. Same as previous right camera matrix of stereo pair will be obtain by R_s and T_s
    # 4. Reprojection error of all left views and its reprojected 3D points per view will be followed by right views 
    # not as before where respective 3D points is reprojected for left and right view in Jacobian matrix

# """
# We are going to first find the 2d matched points across all images. 
# And then calculate the Rli and Tli of only left images of stereo pair, by E decomposition in each i view.
# Use the fix R and T of stereo setup to define the rotation b/w left and right stereo pair.

# Obtain projection matrix for each image of each view. Then using all 2D match of one same feature in all 
# left and right image and along different view compute the 3D world point of that feature.
# """
import os

import sys

PWD = os.getcwd()
sys.path.append(rf'{PWD}/Code')
sys.path.append(rf'{PWD}/Dataset')

print(sys.path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style("darkgrid")
import cv2

from glob import glob
from time import time
from pprint import pprint

from random import sample
import shutil
from calibUtils import calibrate_fisheye, tprint


import cv2 as cv

from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
import time

# To create a webpages
import streamlit as st

def plot_CheckerBoard_corners(path, corners, flag, idx):
    image = cv2.imread(path)
    plt.imshow(image)   
    plt.scatter(np.ravel(corners[0][:])[::2], np.ravel(corners[0][:])[1::2])
    if flag==0:
        plt.title(f"{idx} - Left Image - Detected Corners")
    elif flag==1:
        plt.title(f"{idx} - Right Image - Detected Corners")
    plt.show()
    
def get_camera_intrinsic_params(parameters):
    K = parameters[1]
    K[0][1] = 0
    return K


def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3,4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []
    mean = 0

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.float32(np.matmul(P, pt_3d))
        reprojected_pt /= reprojected_pt[2]

        # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])
        error = cv.norm(pt_2d, reprojected_pt[0:2], cv.NORM_L2)
        mean += error
    
    return (mean/len(points_3d))**0.5

## Find Checkerboard Corners

st.title("Stereo-capture multi-view camera calibration")
st.text("It will take stereo images from multi-view of the checkerboard pattern to estimate the stereo camera parameters.")

st.write("### Provide the path of directory where calibration pattern images are stored")

choices_dataset = ["Yes", "No"]
default_choice_dataset = "No"

choice_dataset = st.radio("Would you like to use your dataset", choices_dataset, 
                  index=choices_dataset.index(default_choice_dataset))

imagesL, imagesR = [], []    #Image paths
camera_ip = ["192.168.1.72", "192.168.1.69"]

if choice_dataset == "No":
    datasets = ["Zhang_Dataset", "Dataset_1", "Dataset_2", "Dataset_3", "Dataset_5"]
    selected_dataset = st.selectbox("Select a dataset", datasets)
    st.write(f"You selected: {selected_dataset}")
    
    if selected_dataset == "Zhang_Dataset":
        path_left = rf"{PWD}/Dataset/{selected_dataset}/left"
        path_right = rf"{PWD}/Dataset/{selected_dataset}/right"

        print(path_left)
        
        imagesL_name = os.listdir(path_left)
        imagesL = [os.path.join(path_left, imgName) for imgName in imagesL_name]
        # st.write(imagesL)
    
        imagesR_name = os.listdir(path_right)
        imagesR = [os.path.join(path_right, imgName) for imgName in imagesR_name]
    
    else:
        imagesL = glob(rf"{PWD}/Dataset/{selected_dataset}/image_*_{camera_ip[0]}.jpg")
        imagesR = glob(rf"{PWD}/Dataset/{selected_dataset}/image_*_{camera_ip[1]}.jpg")
        
else:
    left, right = st.columns(2)
    path_left = left.text_input("Left camera")
    path_right = right.text_input("Right camera")
    
    
    choices = ["Zhang Dataset", "Stereo captured dataset"]
    default_choice = "Zhang Dataset"
    
    choice = st.radio("Select the type of dataset", choices, index=choices.index(default_choice))
    st.text(choice)
        
    if choice == "Zhang Dataset":
        imagesL_name = os.listdir(path_left)
        imagesL = [os.path.join(path_left, imgName) for imgName in imagesL_name]
        # st.write(imagesL)
    
        imagesR_name = os.listdir(path_right)
        imagesR = [os.path.join(path_right, imgName) for imgName in imagesR_name]
    
    if choice == "Stereo captured dataset":
        imagesL = glob(rf"{path_left}/image_*_{camera_ip[0]}.jpg")
        imagesR = glob(rf"{path_right}/image_*_{camera_ip[1]}.jpg")
        # st.write(imagesL)
    
print(imagesL)


st.write("### Furnish some further information for calibration")

ckbrd, sqsz, dim = st.columns(3)
CHECKERBOARD = ckbrd.text_input("Checkerboard (Cols-1, Rows-1) (Eg. (9,6)):").split(',')
squareSize = sqsz.text_input("Square size (in cm) (Eg. 5):")
DIM = dim.text_input("Image dimension (Eg. (640,480)):").split(',')

cal_setting = {"CHECKERBOARD": tuple(CHECKERBOARD), 
               "squareSize": squareSize, 
               "DIM": tuple(DIM) }

# Zhang Default
# CHECKERBOARD = (9,6)
# squareSize = 5
# DIM = (640,480)

# Stereo calibration pattern
# CHECKERBOARD = (14,9)
# squareSize = 7
# DIM = (1920, 1200)

# st.json(cal_setting)
# for k in cal_setting.values():
#     st.text(int(k[0]))

CHECKERBOARD = (int(CHECKERBOARD[0]), int(CHECKERBOARD[1]))
squareSize = int(squareSize)
DIM = (int(DIM[0]), int(DIM[1]))

CORNERS = CHECKERBOARD[0] * CHECKERBOARD[1]



# CHECKERBOARD = (14,9)
# CORNERS = CHECKERBOARD[0] * CHECKERBOARD[1]
# squareSize = 7
# DIM = (1920, 1200)

# Dataset at only 41-70 cm infront of calibration pattern
# imagesL = glob(rf"/mnt/Data2/Datatset/Apr23_StereoDataset/checkerBoardPattern/{trans}/image_*_{camera_ip[0]}.jpg")
# imagesR = glob(rf"/mnt/Data2/Datatset/Apr23_StereoDataset/checkerBoardPattern/{trans}/image_*_{camera_ip[1]}.jpg")
# CHECKERBOARD = (14,9)
# CORNERS = CHECKERBOARD[0] * CHECKERBOARD[1]
# squareSize = 7
# DIM = (1920, 1200)

# Zang Dataset
# path= "/mnt/D112ata2/Datatset/ZhangData/left/"
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


# parameters = \
#         cv2.stereoCalibrate(
#             objpoints,
#             imgpointsL, imgpointsR,
#             K_L, D_L,
#             K_R, D_R,
#             DIM,
#             flags=cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_USE_INTRINSIC_GUESS,
#             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
#         )

parameters = \
        cv2.fisheye.stereoCalibrate(
            objpoints,
            imgpointsL, imgpointsR,
            K_L, D_L,
            K_R, D_R,
            DIM,
            flags=cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e1),
        )
     
# For 17/calib/000 - Change 1e-6 to 1e1.
# The cv::TermCriteria object specifies the termination criteria for the 
# iterative optimization algorithm used by stereoCalibrate. 
# It has three components: the maximum number of iterations, 
# the maximum change in the parameters, and the minimum error tolerance. 
# The abs_max < threshold assertion failure in the error message you received 
# indicates that the maximum change in the parameters has exceeded the specified threshold.

st.write("### Camera parameters by inbuilt calibration (Zhang's Method)")
para1, para2, para3 = st.columns(3)

print("\n\nIntrinsic matrix: ")
K = get_camera_intrinsic_params(parameters)
print("###K####\n\n", K)
para1.subheader("K")
para1.table(K)


print("\n\nStereo setup extrinsic parameters: ")
R_s = parameters[5]
para2.subheader("R_s")
print("###R####\n\n", R_s)
para2.table(R_s)

T_s = parameters[6]
print("\n###T####\n\n", T_s)
para3.subheader("T_s")
para3.table(T_s)

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
print(left_img_idx)

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
    
    
    # F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC)
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

    retval, R_l_temp, t_l_temp, mask = cv.recoverPose(E, pts1, pts2, K)
    
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
    print(f"\n\n{int(view/interval)}: The projection matrix 2 \n" + str(P2))
    
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

# Basics

# Linear mapping that maps a 3-dimensional vector:
# v=⎛⎝⎜v1 v2 v3⎞⎠⎟

# to a corresponding skew-symmetric matrix:
# V=⎛⎝⎜0  -v3  v2 
#      v3  0   -v1 
#     -v2  v1  0⎞⎠⎟

    
# To convert vector into skew-symmetric matrix

def vector_to_skew_symmetric(vector):
    if len(vector) != 3:
        raise ValueError("Vector should be 3 dimensional")
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


# By SVD decomposition 3D point estimation
# Singular-value decomposition

common_3D_pts_svd = []
# This for number of 2d matched points along all views that are 126
for i in range(len(pts_2d_left[0])):
#   This is for number of views
    A = []
    for j in range(len(projectionMatrices_L_before)):
        # Using the rotation matrix from sfm of left and right images after \
        # the bundle adjustment
        P_l = projectionMatrices_L_before[j]
        P_l = np.matmul(K, P_l)
        P_r = projectionMatrices_R_before[j]
        P_r = np.matmul(K, P_r)
        
        # ith 2D point in jth view
        pts_2d_l = np.append(pts_2d_left[j][i], 1)
        pts_2d_r = np.append(pts_2d_right[j][i], 1)
        # Now convert them into skew symm mcommon_3D_ptsatrix and multiply with P_l and P_r respectively
        # Finaaly append in the A matrix for every view
        # Last compute 3D point by last colun of (AT A )-1 * AT
        # np.concatenate( (A, np.dot( vector_to_skew_symmetric(pts_2d_l), P_l)), axis=0)
        # np.concatenate( (A, np.dot( vector_to_skew_symmetric(pts_2d_r), P_r)), axis=0)
        A.extend(np.dot( vector_to_skew_symmetric(pts_2d_l), P_l))
        A.extend(np.dot( vector_to_skew_symmetric(pts_2d_r), P_r))
        
    # common 3d point for all matched feature along all views and stereo pair 
    A = np.array(A)
    U, s, VT = svd(A)
    # Solution for Ax = 0 
    # The possible solution for x will be the column of the V matrix corresponding to 
    # the smallest singular value of s matrix.
    # print(U.shape)
    # print(s.shape, np.argmin(s) )
    # print(VT)
    # We can take the column corresponding to np.argmin(s) of VT.T or row of VT would also work 
    # print(VT[np.argmin(s)]
    # common_3D_pts_svd.append( VT[:, np.argmin(s)] )
    common_3D_pts_svd.append( VT[np.argmin(s)] )
    
    # Not a correct way for the Ax=0 solution      
    # common_3D_pts_svd.append( np.dot( np.dot( np.linalg.inv(np.dot(A.T, A)), A.T), np.ones(A.shape[0])) )

common_3D_pts_svd = np.array(common_3D_pts_svd).T
common_3D_pts_svd /= common_3D_pts_svd[3]
common_3D_pts_svd = np.array(common_3D_pts_svd).T

# Display 3D points before and after bundle adjustment on streamlit page
beforeBA_3d, afterBA_3d = st.columns(2) 

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig)

x_ = []
y_ = []
z_ = list()

for p in common_3D_pts_svd:
    x_.append(p[0])
    y_.append(p[1])
    z_.append(p[2])
        
    
# ax.scatter(2,3,4) # plot the point (2,3,4) on the figure
ax.scatter(x_, y_, z_)

# setting title and labels
ax.set_title("Obtaned 3D points by trianulation from all views")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()
beforeBA_3d.pyplot(fig)

#Stereo Camera calibration - MultiView Images

st.write("### Projecting 3D common points in all views")
st.text("Note: 0th view is the refrence view with P= K[I 0]")
# Ploting 2D detected corners as well as reprojection of 3D points on image
def reprojection_3D_pt(path, corners, flag, pos, pts_3D, K, P):
    
    # Detected corners
    image = cv2.imread(path)
    # plot the image and corners
    fig, ax = plt.subplots()
    ax.imshow(image)
    # Actual detected corners
    ax.scatter(np.ravel(corners)[::2], np.ravel(corners)[1::2], 
               color = 'blue', marker = '.', label="Detected corners")
    
    # Ploting 3D points reprojection
    a=[]
    for idx, pt_3d in enumerate(pts_3D):
        reprojected_pt = np.float32(np.matmul(np.matmul(K, P), pt_3d))
        reprojected_pt /= reprojected_pt[2]
        a.append(reprojected_pt[0:2])
    
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    # for i in range(len(a)):  
    #     ax.scatter(a[i][0], a[i][1], color='red', marker="x")
    ax.scatter(np.ravel(a)[::2], np.ravel(a)[1::2], 
               color = 'red', marker = 'x', label="Re-projected common 3D points")
    if flag ==0:
        ax.set_title(f"{pos} Left Image")
    elif flag==1:
        ax.set_title(f"{pos} Right Image")
    # display the plot using st.pyplot()
    ax.legend()
    # st.pyplot(fig)
    fig.set_size_inches(3, 3)  # 8 inches wide by 6 inches high
    return fig

# Inside iPython console in preference
# Change the Graphics to Auto or inline

# I am using points_3d via a triangulation on one stereo pair match points  
# As common_3D_pts rather then by linear equation solution because
# They are giving better reprojected point along views and the common_3D_pts 3D plot is not good as well
common_3D_pts = common_3D_pts_svd

for view in range(len(left_img_idx)-1):
    
    P_l = projectionMatrices_L_before[view]
    P_r = projectionMatrices_R_before[view]

    idx = left_img_idx[view+1]
    leftImg, rightImg = st.columns(2)
    
    figL = reprojection_3D_pt(path = imagesL[idx], corners = pts_2d_left[view][:], flag=0, pos=idx, 
                       pts_3D = common_3D_pts, K = K, P = P_l)
    leftImg.pyplot(figL)
    
    # Similarily for right image
    figR = reprojection_3D_pt(path = imagesR[idx], corners = pts_2d_right[view][:], flag=1, pos=idx, 
                       pts_3D = common_3D_pts, K = K, P = P_r)
    rightImg.pyplot(figR)

#%% Selecting only good views - threshold of 50 on per point rpe in an image per view

max_rpe_views = [] 
final_view = []

st.write("### We select the views with least RPE error for bundle adjustment")

choices_threshold = ["Yes", "No"]
default_choice_threshold = "No"

choice_threshold = st.radio("Would you like to enter value for threshold", choices_threshold, 
                  index=choices_threshold.index(default_choice_threshold))
st.text(choice_threshold)

THRESHOLDS = {"Zhang_Dataset": 250, 
              "Dataset_1": 500, 
              "Dataset_2": 900, 
              "Dataset_3": 4000, 
              "Dataset_5": 9000}

threshold = 0

if choice_threshold == "No":
    threshold = THRESHOLDS[selected_dataset]
else:
    threshold = st.text_input("Enter the threshold value for max RPE in a view: ")
    threshold = int(threshold)
st.write(f"####Value of threshold is {threshold}")

for pos in range(len(projectionMatrices_L_before)):
    
    P_l = projectionMatrices_L_before[pos]
    P_r = projectionMatrices_R_before[pos]
    
    P_lr = [P_l, P_r]
    # Since Iref = 0th instance of left_img_idx, for that P_l and P_r are not calculated
    # corner_2D_lr = [imgpointsL[view+1][0], imgpointsR[view+1][0]]   #See view =0 is reference frame while P_l estimation
    corner_2D_lr = [pts_2d_left[pos], pts_2d_right[pos] ]
    path_lr = [imagesL, imagesR]
    rev_idx = 0
    
    a=[]  
    for P, corner_2D in zip(P_lr, corner_2D_lr):
        # # #To view detected corner and reprojected points for debugging purpose
        # path = ''
        # tag =''
        # if rev_idx == 0:
        #     path = path_lr[0][left_img_idx[pos+1]] 
        #     Image view = left_img_idx[pos+1]
        #     # As left_img_idx[0] is the reference frame
        #     # for which there is no P and 2D points for it are stored
        #     tag = 'Left'

        # else:
        #     path = path_lr[1][left_img_idx[pos+1]]
        #     tag = 'Right'
            
        for idx, pt_3d in enumerate(common_3D_pts):
            reprojected_pt = np.float32(np.matmul(np.matmul(K, P), pt_3d))
            reprojected_pt /= reprojected_pt[2]
            
            rpe_point =  np.array(corner_2D[idx] - reprojected_pt[0:2]).ravel()
            # print(idx, rpe_point)
            # print(left_img_idx[pos+1], rpe_point, end="\n\n")
            a.extend(rpe_point)
            
        #     # #To view detected corner and reprojected points for debugging purpose
        #     # print(path)
        #     image = cv2.imread(path)
        #     plt.imshow(image)
        #     plt.scatter(corner_2D[idx][0], corner_2D[idx][1], color='blue')
        #     plt.scatter(reprojected_pt[0], reprojected_pt[1], color='red')
        #     plt.title(f"{tag} View - {left_img_idx[pos+1]} and Point - {idx} - Point - {reprojected_pt[:2]}")
        #     plt.show()
        # rev_idx = ~rev_idx
        
    print(max(abs(np.array(a))))
    max_rpe_views.append( round(max(abs(np.array(a))), 2) )
    
    if max(abs(np.array(a))) < threshold:
        final_view.append(left_img_idx[pos+1])

df = {"View": range(1, len(max_rpe_views)+1),
      "Max RPE per view": max_rpe_views}

df = pd.DataFrame(df)
df = df.reset_index(drop=True)

st.write("#### Maximum RPE among all points in a views:")
st.dataframe(df, width=300, height=300)

st.write("**After thresholding the selected views are:**")
print(final_view)
st.text(final_view)
#%% Bundle Adjustment 

#BA n View - I

# """
# Created on Mon Feb  6 15:28:51 2023

# @author: lakshayb

# # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

# 2. https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
# """


from scipy.optimize import least_squares

# least_squares calculate the residuals for each number of unknown
# The dimension of unknown correspond to the number of column of J 
# Rows of J, equal to the calculated residuals, which are num of match feature across all 
# views * 2 * num of corners, 2 correspond the x and y pixel location as residual
# is between projected and reprojeected points (pt_2d - reprojected_pt[0:2])

# For each iteration least_square function calculate the all residuals (That correspond to rows of J) 
# equal to number of columns of J (num of unknown) by 
# varying the unkwown for each run wrt individual unknown

def get_intrinsics(vals):
    # just optimise f
    f = vals[0]
    K = np.eye(3)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = vals[1]
    K[1, 2] = vals[2]
    return K

def reprojection_loss_function(opt_variables, pts_2d_left, pts_2d_right, num_pts, K):
    '''
    opt_variables --->  K, R_s, T_s, Camera Projection matrix of left views of stereo pair + All 3D points
    '''
    # Decomposing K, R_s and T_s of the stereo setup
    # We just pass the focal length, o_x and o_y to form the K matrix by get_intrinsics function
    K2 = get_intrinsics( opt_variables[0:3] )
    
    R_s = opt_variables[3:6]
    R_s = cv2.Rodrigues(R_s)[0]
    
    C_s = opt_variables[6:9].reshape((3,1))
    T_s = -1*np.matmul(R_s, C_s)
    
    R_L = opt_variables[9:12]   # Would be Identity matrix only
    R_L = cv2.Rodrigues(R_L)[0]
    
    nViews= int( (len(opt_variables) -12 -num_pts*3) / 6 )  # 6 = Unkown of P matrix of left views
    # print(nViews)
    
    P_lr = opt_variables[12: 12 + 6*nViews].reshape(nViews,2,3) 
    # 6*nViews*2 - Parameter of all left view followed by right view parameters
    P_l = P_lr
    # P_l = P_lr[0:nViews]
    # P_r = P_lr[nViews:]
    
    point_3d = opt_variables[12 + 6*nViews:].reshape((num_pts, 3))
    
    # Now append the array of residual that will used in the bundle adjustment
    rep_error = []
    # print(np.array(rep_error).shape)
    
    # For all left views
    for k in range(nViews):
        # Converting Rodrigues angle to rotation matrix
        r_L_matrix , _ = cv2.Rodrigues( P_l[k][0] )
        C_L = P_l[k][1].reshape((3, 1))
        P = np.hstack([r_L_matrix, -1*np.matmul(r_L_matrix, C_L)])
        
        dummy = np.hstack([R_L, np.zeros((3,1)) ])
        dummy2 = np.vstack([P, [0, 0, 0, 1]])
        
        # As R_L which is identity matrix remains unchanged in BA process
        # P = np.matmul(dummy, dummy2)
        Pl = np.matmul(K, P)
        
        points_2d = pts_2d_left[k]

        for idx, pt_3d in enumerate(point_3d):
            # print(pt_3d.shape)
            # Kth view, xth camera and jth matched feature 
            pt_2d = np.array([points_2d[idx][0], points_2d[idx][1]])
            reprojected_pt = np.matmul(Pl, np.hstack([pt_3d, 1]))
            reprojected_pt /= reprojected_pt[2]
            # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
            rep_error.append((pt_2d - reprojected_pt[0:2])*112)
    
    # For all right views
    for k in range(nViews):
        # Converting Rodrigues angle to rotation matrix
        r_L_matrix , _ =	cv2.Rodrigues( P_l[k][0] )
        C_L = P_l[k][1].reshape((3, 1))
        P = np.hstack([r_L_matrix, -1*np.matmul(r_L_matrix, C_L)])
        
        dummy = np.hstack([R_s, T_s])
        dummy2 = np.vstack([P, [0, 0, 0, 1]])
        
        P = np.matmul(dummy, dummy2)
        Pr = np.matmul(K, P)
        
        points_2d = pts_2d_right[k]

        for idx, pt_3d in enumerate(point_3d):
            # print(pt_3d.shape)
            # Kth view, xth camera and jth matched feature 
            pt_2d = np.array([points_2d[idx][0], points_2d[idx][1]])
            reprojected_pt = np.matmul(Pr, np.hstack([pt_3d, 1]))
            reprojected_pt /= reprojected_pt[2]
            # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
            rep_error.append((pt_2d - reprojected_pt[0:2])*112)
    
    # print(np.array(rep_error).shape)
    return np.array(rep_error).ravel()


def bundle_adjustment(common_3D_pts, pts_2d_left, pts_2d_right, P_BA, K):
    
    opt_variables = np.hstack((P_BA, common_3D_pts.ravel()))
    # print(opt_variables.shape)
    num_points = len(common_3D_pts)
    # print(num_points)
    
    # It prints the losses at each iteration
    # corrected_values = least_squares(reprojection_loss_function, opt_variables, args=(points_2d,num_points), verbose=2)
    
    # For stereo dataset 1, which is linearly at equal distance 
    # corrected_values = least_squares(reprojection_loss_function, opt_variables, args=(pts_2d_left, pts_2d_right,num_points), verbose=2, ftol=1e-03, xtol=1e-05, gtol=1e-05)
    # corrected_values = least_squares(reprojection_loss_function, opt_variables, args=(pts_2d_left, pts_2d_right,num_points), verbose=2, ftol=1e-08, xtol=1e-10)
    
    corrected_values = least_squares(reprojection_loss_function, 
                                     opt_variables, 
                                     args=(pts_2d_left, pts_2d_right,num_points, K), 
                                     verbose=2, x_scale='jac', ftol=1e-7, xtol=1e-10, method='lm')
    # x_scale='jac',
    # Without any output from the least square function
    
    # max_nfevNone or int, optional
    # Maximum number of function evaluations before the termination. 
    # If None (default), the value is chosen automatically:
    # For ‘trf’ and ‘dogbox’ : 100 * n. 
    # For ‘lm’ : 100 * n if jac is callable and 100 * n * (n + 1) otherwise 
    # (because ‘lm’ counts function calls in Jacobian estimation).
    # where n is the number of unknown in J matrix or no. of column
    
    # method{‘trf’, ‘dogbox’, ‘lm’}, optional with default being 'trf'
    # ‘lm’ : Levenberg-Marquardt algorithm needs to be performed
    # corrected_values = least_squares(reprojection_loss_function, opt_variables, args=(pts_2d_left, pts_2d_right,num_points, K), verbose=2, ftol=1e-7, xtol=1e-7, gtol=1e-10)

    # corrected_values = least_squares(reprojection_loss_function, opt_variables, args=(pts_2d_left, pts_2d_right,num_points, K), verbose=2, ftol=1e-7, xtol=1e-7)
    # reprojection_error = reprojection_loss_function(opt_variables, points_2d, num_pts)

    # print("The optimized values \n" + str(corrected_values))
    nViews= int( (len(opt_variables) -12 -num_points*3) / 6 )  # 6 = Unkown of P matrix
    P = corrected_values.x[0:12 + 6*nViews]
    points_3d = corrected_values.x[12 + 6*nViews:].reshape((num_points, 3))

    return P, points_3d, corrected_values



# Appending first the left camera all view P
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
      

# BA n View - II

# Now to perform BA along all views and taking P of left and Right camera
# we need to pass the list conatining all the P matrix and then 
# Need to calculate the residual for all the P and 3D points

st.write("##### Performing bundle adjustment")
print("Performing bundle adjustment")
t0 = time.time()
# 2. 3D points will be used in cartesian coordinate form then in homogeneous coordinates - 3 Parameter durin BA
P_after_BA, common_3D_pts_after_BA, corrected_values = bundle_adjustment(
    np.array(common_3D_pts_svd[:, :3]), 
    np.array(pts_2d_left)[list((np.array(final_view)/interval).astype('int') -1)], 
    np.array(pts_2d_right)[list((np.array(final_view)/interval).astype('int') -1)], 
    P_BA, K)    

t1 = time.time()
st.text("It took {0:.0f} seconds to perform bundle adjustment".format(t1 - t0))
st.balloons()
print("Optimization took {0:.0f} seconds".format(t1 - t0))


# Now ploting the corrected 3D points after BA and reprojected points from BA

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig)

x_temp = []
y_temp = []
z_temp = list()

for p in common_3D_pts_after_BA:
    x_temp.append(p[0])
    y_temp.append(p[1])
    z_temp.append(p[2])
        
    
# ax.scatter(2,3,4) # plot the point (2,3,4) on the figure
ax.scatter(x_temp, y_temp, z_temp)

# setting title and labels
ax.set_title("3D points after bundle adjustment")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()
afterBA_3d.pyplot(fig)

"""_________________________________________________________"""
# Seperating the P_l_afterBA and P_r_afterBA

K_after_BA = get_intrinsics( P_after_BA[0:3] )

R_after_BA = cv2.Rodrigues( P_after_BA[3:6] )[0]
C_after_BA = P_after_BA[6:9].reshape(3,1)
T_after_BA = -1*np.matmul(R_after_BA, C_after_BA)

R_L_after_BA = cv2.Rodrigues( P_after_BA[9:12] )[0]

nViews= int( (len(P_after_BA) - 12) / 6 ) # -15 because 15 parameter of stereo setup K, R_s and T_s

# P_after_BA = P_after_BA[15:].reshape(nViews,2,3)

P_l_afterBA = P_after_BA[12:].reshape(nViews,2,3)   # As at 0 now have [R T] the setereo setup 
# P_r_afterBA = P_after_BA[nViews:]


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
    
    leftImg, rightImg = st.columns(2)
    
    # print(i)
    figL = reprojection_3D_pt(path = imagesL[fn_view], corners = pts_2d_left[i-1][:], flag=0, pos=fn_view, 
                       pts_3D = common_3D_pts_after_BA, K = K_after_BA, P = P_l)
    leftImg.pyplot(figL)

    # Similarily for right image
    figR = reprojection_3D_pt(path = imagesR[fn_view], corners = pts_2d_right[i-1][:], flag=1, pos=fn_view, 
                       pts_3D = common_3D_pts_after_BA, K = K_after_BA, P = P_r)
    rightImg.pyplot(figR)

#%% Error Analysis
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


st.dataframe(rpe)

# Average RPE

sum_rpe = np.sum(np.array(rpe['beforeBA']))
length = len(rpe['beforeBA'])
print("Average re-projection error before BA: ", sum_rpe / length)
st.write(f"##Average re-projection error before BA: {sum_rpe / length}")

sum_rpe = np.sum(np.array(rpe['afterBA']))
length = len(rpe['afterBA'])
print("Average re-projection error after BA: ", sum_rpe / length)
st.write(f"##Average re-projection error after BA: {sum_rpe / length}")

#%% Projecting points

# Plotting single detected corner and reproject point after BA in image

for pos, view in enumerate(final_view):
    
    # P_l = P_l_afterBA[view]
    # P_r = P_r_afterBA[view]
    
    # P_l = projectionMatrices_L_before[view]
    
    r_L_matrix , _ = cv2.Rodrigues( P_l_afterBA[pos][0] )
    C_L = P_l_afterBA[pos][1].reshape((3, 1))
    P = np.hstack([r_L_matrix, -1*np.matmul(r_L_matrix, C_L)])
    
    dummy = np.hstack([R_L_after_BA, np.zeros((3,1)) ])
    dummy2 = np.vstack([P, [0, 0, 0, 1]])
    P_l = np.matmul(dummy, dummy2)
    
    # dummy = np.hstack([R_after_BA, T_after_BA])
    # dummy2 = np.vstack([P_l, [0, 0, 0, 1]])
    
    # P_r = np.matmul(dummy, dummy2)
    
    # P_LR = [P_l, P_r]
    # points_2d_LR = [pts_2d_left[view], pts_2d_right[view]]
    
    # for P, points_2d in zip(P_LR, points_2d_LR):
    #     # print(points_2d.shape)
    error = 0
    for idx, pt_3d in enumerate(common_3D_pts_after_BA):
        # print(pt_3d.shape)
        # Kth view, xth camera and jth matched feature 
        pt_2d = np.array([pts_2d_left[view-1][idx][0], pts_2d_left[view-1][idx][1]])
        reprojected_pt = np.float32(np.matmul(np.matmul(K, P_l), pt_3d))
        reprojected_pt /= reprojected_pt[2]
        
        error += np.sum((pt_2d - reprojected_pt[0:2])**2)
        # print(f"{idx} - {error}")

        # Ploting Image
        # path_idx = left_img_idx[view+1]
        # path = imagesL[path_idx]
        path = imagesL[view]
        image = cv2.imread(path)
        plt.imshow(image)
        plt.scatter(pt_2d[0], pt_2d[1], color='red')
        plt.scatter(reprojected_pt[0], reprojected_pt[1], color='blue')
        plt.title(f"{idx} - Point - {reprojected_pt[:2]}")
        # if flag ==0:
        #     plt.title(f"{pos} Left Image - Detected Corners")
        # elif flag==1:
        #     plt.title(f"{pos} Right Image - Detected Corners")
        plt.show()
    break


#%% Fitting a plane to the obatined 3D points after BA

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig)

x_ = []
y_ = []
z_ = list()

for p in common_3D_pts_after_BA:
    x_.append(p[0])
    y_.append(p[1])
    z_.append(p[2])
        
    
# ax.scatter(2,3,4) # plot the point (2,3,4) on the figure
ax.scatter(x_, y_, z_)

# setting title and labels
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')



points = np.array(common_3D_pts_after_BA)

# Solve for the null space of the matrix formed by the points
u, s, vh = np.linalg.svd(points)
plane_coeffs = vh[-1]

# The plane equation in homogeneous coordinates is given by:
# ax + by + cz + d = 0
a, b, c, d = plane_coeffs

# Normalize the plane equation coefficients
norm = np.sqrt(a**2 + b**2 + c**2)
a /= norm
b /= norm
c /= norm
d /= norm

# Print the plane equation
print(f"The plane equation is: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

ax.set_title(f"Equation of a plane is: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
plt.show()
st.pyplot(fig)
