#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:16:27 2023

@author: lakshayb
"""

import streamlit as st
import matplotlib.pyplot as plt

st.title("Literature Review")

st.write('''
The first task in camera calibration is to capture images of the calibration pattern.
During data acquisition, either the calibration pattern or camera module or both
moved w.r.t. each other, to diversify the view in capture images. Camera motion [1]
while capturing the calibration pattern can make temporary deformations which
can introduce significant errors in estimated camera parameters. The effect of 
deformation can be lowered by a low-dimensional deformation model with only a few
parameters per image.\n
Various monocular camera calibration techniques reduce a nonlinear error function, 
but at the expense of longer computation times and need a more powerful
processor. Closed-form solutions, however, have also been put forth (e.g., [5]). \n
However, because these techniques rely on a simplified camera model, their performance 
falls short of nonlinear minimization.
Stereo camera calibration employs the fundamental matrix to calculate both intrinsic and extrinsic parameters. 
A good estimation of the fundamental matrix determines the reliability of the computed parameters. 
The corresponding matched feature in the left and right images captured by 
the stereo camera system determines how to estimate the fundamental matrix. With stereo cameras that are only
translated horizontally, the stereo image reduces the search space of the matched
feature in the right image in the horizontal direction. While due the presence of
noise in the image-capturing system and rotation between stereo cameras increase the search space to 
the plane which in turn reduces the correctness of matched feature. There are various methods exist for 
the estimation of camera parameters from stereo camera systems.\n
Huang and Mitchell [6] present a dynamic calibration with a stereo camera setup
using a two-step camera calibration where the first step is to estimate the camera
intrinsic parameters using the absolute orientation of an object, and the second step
estimates the relative orientation of stereo cameras by observing an unknown object. 
Over the lifetime of the camera, the camera parameters get changed from the factory calibration. 
Camera calibration can be performed with the help of multicamera and multi-view images [7] 
of different 3D scenes having dense features at
different depths. \n
Hartley et al.[8] presents the estimation of 3D point location without explicit
estimation of camera models given two uncalibrated perspective views. Various
solutions for the 3-D location of points can be estimated that are compatible with
the given set of matched points. They estimated the final 3-D location of the points
utilizing ground control points, which were later employed to fine-tune the camera
parameters. Rothwell et al.[4] compare the five different methods for reconstructing 3D scenes 
based on triangulation on two perspectives and some methods that
impose constrain on the basis of actual measurement in the physical scene. \n
We only need to reconstruct the 3D points that correspond to the matched points
in the left and right images of the calibration objects for calibration purposes. Zhang
et al.[9] computes the essential matrix while integrating additional visual information 
(e.g. matched feature from a stereo pair of face images) in addition to that from
the calibration object. They used the cost function, which is the sum of squared
distances between the reconstructed points and the known 3-D Euclidean points,
rather than minimizing the re-projection error between the re-projected point and
its corresponding detected corners in the image plane.\n
Stereo calibration using two images at different time instances of an orthogonal checkerboard pattern is presented by Zhang et al.[10]. 
The estimated camera matrix was further optimized through non-linear minimization by minimizing the
re-projection error across all four images in a single iteration. Bundle adjustment
[2] refines the 3D structure and camera parameters of a scene reconstructed from
multiple images. In using a stereo camera configuration, it is now necessary to take
into account both left and right images from all perspectives while minimizing the
re-projection error between the observed image points and the projections of the 3D
points onto the images [3].\n
Owing to the vast application of camera calibration, the vehicle speed can be
calculated in real-time using the roadside camera [11]. Using a three-stage algorithm, where the
first step is to estimate the position of the camera w.r.t. roadway
using motion direction and edges of the vehicles. In the second step, they calibrate
the camera by lane boundaries and vanishing point followed by transformation of
an image coordinate of a vehicle in an image to real-world coordinates using the
camera model and then successfully calculated vehicle speed. Suree and Daniel
[12] created a virtual speed sensor using the existing network of the low-quality
roadside camera of the transportation department.
''')
