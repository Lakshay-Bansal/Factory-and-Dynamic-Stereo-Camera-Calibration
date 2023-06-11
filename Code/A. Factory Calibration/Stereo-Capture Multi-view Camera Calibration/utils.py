import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    # K = []
    # with open(calibration_file_dir + '/K_mat.txt') as f:
    #     lines = f.readlines()
    #     calib_info = [float(val) for val in lines[0].split(' ')]
    #     row1 = [calib_info[0], calib_info[1], calib_info[2]]
    #     row2 = [calib_info[3], calib_info[4], calib_info[5]]
    #     row3 = [calib_info[6], calib_info[7], calib_info[8]]

    #     K.append(row1)
    #     K.append(row2)
    #     K.append(row3)
    
    # K = np.loadtxt("..\K_mat.txt")
    # K = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]])
    # K = np.array([
    # [789.7377781,	0,	950.939424],
    # [0,	790.2364756,	614.2075994],
    # [0, 0, 1]])
    K = parameters[1]
    K[0][1] = 0
    # print("K Matrix: ", K)
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
        error = cv2.norm(pt_2d, reprojected_pt[0:2], cv2.NORM_L2)
        mean += error
    
    return (mean/len(points_3d))**0.5

def calibrate_fisheye(all_image_points, all_true_points, image_size, K, D, FLAGS, verbose=0):
#    print(all_true_points.shape, type(all_true_points))
    while True:
        assert len(all_true_points) > 0, "There are no valid images from which to calibrate."
        try:
            rms, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=all_true_points,
                imagePoints=all_image_points,
                image_size=image_size,
                K=K,
                D=D,
                flags=FLAGS,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            )
#            print('\t\tFound a calibration based on {} well-conditioned images.'.format(len(all_true_points)))
            return rms, mtx, dist, rvecs, tvecs
        except cv2.error as err:
            if(verbose==1):print(err)
            try:
                idx = int(str(err).split()[-4])  # Parse index of invalid image from error message
#                print(idx)
                all_true_points = np.delete(all_true_points,idx, 0)
                all_image_points = np.delete(all_image_points,idx, 0)
            except:
                raise err

def tprint():
    print("test print function!")
    print(cv2.__version__)


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
    plt.show()

# Inside iPython console in preference
# Change the Graphics to Auto or inline


#Fitting a plane to the obatined 3D points after BA
def fit_3D_plane(common_3D_pts_after_BA):
    fig = plt.figure(figsize=(4,4))
    ax = Axes3D(fig)

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

    ax.set_title(f"equation is: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    plt.show()