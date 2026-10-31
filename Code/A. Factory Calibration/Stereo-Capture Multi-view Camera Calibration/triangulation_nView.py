import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D


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

# Example usage
# v = np.array([1, 2, 3])
# skew_matrix = vector_to_skew_symmetric(v)
# print(skew_matrix)e = cv2.imread(imagesL[left_img_idx[view+1]])


# ---------------------------------------------------------------------------------------------------------------------#
# Wrong Way of common 3D points from all view image
# common_3D_pts = []
# # This for number of 2d matched points along all views that are 126
# for i in range(len(pts_2d_left[0])):
# #   This is for number of views
#     A = []
#     for j in range(len(projectionMatrices_L_before)):
#         # Using the rotation matrix from sfm of left and right images after \
#         # the bundle adjustment
#         P_l = projectionMatrices_L_before[j]
#         P_r = projectionMatrices_R_before[j]
        
#         # ith 2D point in jth view
#         pts_2d_l = np.append(pts_2d_left[j][i], 1)
#         pts_2d_r = np.append(pts_2d_right[j][i], 1)
#         # Now convert them into skew symm mcommon_3D_ptsatrix and multiply with P_l and P_r respectively
#         # Finaaly append in the A matrix for every view
#         # Last compute 3D point by last colun of (AT A )-1 * AT
#         # np.concatenate( (A, np.dot( vector_to_skew_symmetric(pts_2d_l), P_l)), axis=0)
#         # np.concatenate( (A, np.dot( vector_to_skew_symmetric(pts_2d_r), P_r)), axis=0)
#         A.extend(np.dot( vector_to_skew_symmetric(pts_2d_l), P_l))
#         A.extend(np.dot( vector_to_skew_symmetric(pts_2d_r), P_r))
        
#     # common 3d point for all matched feature along all views and stereo pair 
#     A = np.array(A)
#     common_3D_pts.append( np.dot( np.dot( np.linalg.inv(np.dot(A.T, A)), A.T), np.ones(A.shape[0])) )
# ---------------------------------------------------------------------------------------------------------------------#

def triangulation_nView(pts_2d_left, pts_2d_right, projectionMatrices_L_before, projectionMatrices_R_before, K):
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

    fig = plt.figure(figsize=(4,4))
    ax = Axes3D(fig)

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
    ax.set_title("Obtaned 3D points by SVD")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    plt.show()
    
    return common_3D_pts_svd