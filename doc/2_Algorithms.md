# About

Project explore the estimation of the camera parameters such as intrinsic (focal length, camera offset) and 
extrinsic parameters (rotation and translation of stereo setup) of the stereo camera.

We can classify camera calibration in two broad types:

1. Factory Calibration
2. Dynamic Calibration

<p align="center">
  <img src="https://github.com/Lakshay-Bansal/Dynamic-Stereo-Camera-Calibration/assets/84389082/aba12f43-9f36-4519-a171-5e887781840d.png" /><br>
  <b>Figure: Types of camera calibration.</b><br>
</p>

1. Factory calibration - Camera parameters are estimated by using checkerboard pattern so as to make 3D measurement (depth) more accurate by performing bundle adjustment.

2. Dynamic calibration - Calibration using real scene images for correcting the possible deformation due to
transportation and handling.

# A. Factory Camera Calibration

## Stereo-Capture Multi-view Camera Calibration

This algorithm is performed on the checkerboard pattern. The sequential steps involved in stereo capture multi-view camera calibration are depicted in Figure 1.

<p align="center">
  <img src="https://github.com/Lakshay-Bansal/Dynamic-Stereo-Camera-Calibration/assets/84389082/f465638b-41d0-468a-ad6b-65116be0e563.png" /><br>
  <b>Fig. 1: Steps of stereo capture multi-view camera calibration.</b><br>
</p>



# B. Dynamic Camera Calibration


## 1. Single Capture Dynamic Calibration Algorithm (SCDCA)

This algorithm is performed on the dynamic scene images. Using photographs of the calibration pattern of known dimensions with different
descriptions like the number of rows, the number of columns, and the size of a square in a checkerboard pattern a camera calibration can be carried out. 
This is a dynamic camera calibration performed with images of dynamic outdoor landscapes using a 
single capture algorithm to calibrate the camera. The steps involved in the algorithm are shown in Fig. 2. 

<p align="center">
  <img src="https://github.com/Lakshay-Bansal/Dynamic-Stereo-Camera-Calibration/assets/84389082/ab2cf9e8-9993-42b8-8cbc-5f197598c4f6.png" /><br>
  <b>Fig. 2: Flowchart of single capture dynamic calibration algorithm.</b><br>
</p>

A camera system is used to acquire multi-view images, and then features are detected in the images. The detected features are then mapped to the corresponding features 
in different view images. After doing the essential matrix computation with the help of the 
five-point algorithm approach on the matched feature, 3D world points are obtained from the matched feature. The best 3D points are then
chosen by bundle adjustment, which reduces the re-projection error. Additionally, guided feature matching will be carried out to choose the best feature. The final
step in the camera calibration process is validation in which the 3D measurement of the object in an image can be performed, to confirm that the camera parameters that
were determined using a single capture dynamic camera calibration are accurate.


## 2. Stereo-Capture Multi-view Dynamic Camera Calibration


Fig. 3, presents the flowchart of multi-view dynamic stereo camera calibration
which takes single-capture dynamic calibration algorithm 
as a processing block in some step of the following algorithm. The reference image
is denoted as Iref , which is the most recently captured image frame set for which
dynamic calibration parameters need to be estimated. The pool of N candidate
image frame sets is denoted as Ii∈{1,...,N}.

<p align="center">
  <img src="https://github.com/Lakshay-Bansal/Dynamic-Stereo-Camera-Calibration/assets/84389082/3d360bde-f977-47c3-9a7a-b6ce803a40a8.png" /><br>
  <b>Fig. 3: Flowchart of multi-view dynamic stereo camera calibration.</b><br>
</p>

