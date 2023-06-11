1. BA_stereoscopic3D.py and BA_stereoscopic3D_v2.py are the implemenation of the Paper that talks about stereoscopic bundle adjsutment. In this code itself the Bundle adjustment function are encorporated and modified acordingly.

2.BA_stereoscopic3D_someview.py is the extension of the above version of the code but added the view rejection criteria as well for stereo bundle adjustment on the basis of the re-projection error per point per view as presented in the MATLAB code of stereo bundle adjustment.

_________________________________________________________________________________________________________________________________________________________________________
Note: Mis understood the Camera centre and incorrectly obatin the P_l and P_r of each stereo pair view. Hence as the result the common 3D points after bA is not correct and that's why the reprojected corners as well.


# As writing P_l = K [R_l -R_l t_L], this is not correct because
# Correct P_l = K [R_l -R_l C], where C is the camera centre defined as C = -R_L^-1 t_L
# Hence correct P_l = K [R_l -R_l -R_L^-1 t_L] = K [R_l t_l]
# So the implementation in main_by K E_v2.py is correct implementation 



# As writing P_r = K [R -Rt]|R_l -R_l t_L|
#                           | 0    1     |, this is not correct because
# Correct P_r = K [R -R C]|R_l -R_l C_L|
#                        | 0    1  |, where 
# C is the camera centre defined as C = -R^-1 T and C_L = -R_L^-1 t_L
# Hence correct P_l = K [R T] |R_l t_L|
#                             | 0    1|
# So the implementation in main_by K E_v2.py is correct implementation

# R_r is obatined by the R and T of the camera setup on the R_l_1



