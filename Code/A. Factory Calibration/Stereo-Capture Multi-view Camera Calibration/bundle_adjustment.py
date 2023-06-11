# Bundle Adjustment n View
"""
@author: lakshayb

# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

2. https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html


# This code treat unknown during bundle adjustment:
    # 1. Rather then using complete Rotation matrix we will use its Rodriques angles only - 3 Parameters
    # 2. 3D points will be used in cartesian coordinate form then in homogeneous coordinates - 3 Parameter
    # 3. Right camera matrix of stereo pair will be obtain by R_s and T_s
    # 4. Reprojection error of all left views and its reprojected 3D points per view will be followed by right views 
    # not as before where respective 3D points is reprojected for left and right view in Jacobian matrix
"""

import numpy as np
import cv2
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
                                     verbose=2, x_scale='jac', ftol=1e-5, xtol=1e-7, method='trf')
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