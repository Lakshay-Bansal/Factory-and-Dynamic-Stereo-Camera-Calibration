#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:16:27 2023

@author: lakshayb
"""

import streamlit as st
import matplotlib.pyplot as plt
import cv2

# Function to display image the resize image
def resize(path, scale_factor=0.5):
    image = cv2.imread(path)
    # scale_factor = 0.5  
    # Calculate the new width and height based on the scale factor
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Convert the resized image back to RGB format
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return resized_image_rgb

import os
import sys

PWD = os.getcwd()
sys.path.append(rf'{PWD}\Images')

st.title("Introduction")

st.write('''Camera Calibration refers to the estimation of intrinsic camera parameters like focal 
         length, principal point, the skew of a CCD array, and extrinsic camera parameters
        which account for the relative position of a camera with respect to a world coordinate
         system known as structure from motion (SFM). While for stereo cameras extrinsic camera 
         parameters that are of interest are the rotation and translation of the
        right camera with respect to the left camera. This needs a calibration pattern (e.g.
        checkerboard pattern). On the other hand, dynamic camera calibration is carried
        out with real 3D scene images with dense features at varying depths. Nowadays
        there is a trend of a multi-camera system in consumer devices with functionality
        like portrait mode (depth-based background blur in an image see Fig. 1.1), applications 
        like an estimation of a dimension of an object and its depth as shown in
        Fig. 1.2. ''')

st.image(r'Images/depthBlur.jpg', 
         caption='''Figure 1.1: Portrait mode. Photo from PSD Stack. "Photoshop’s New Depth Blur Filter."
                Source: https://www.psdstack.com/photoshop-tutorials/basic/depth-blur-filter''',
                width=400)
st.image('Images/App_CamCal.png', 
         caption='''Figure 1.2: Object dimension detection in an image. Photo from MATLAB tutorial.
         Source: https://in.mathworks.com/help/vision/ug/camera-calibration.html ''',
                width=500)

st.write('''In planetary explorations (eg. the Mars mission), the stereo-vision system needs
to be strongly calibrated in order to perform a 3-D Euclidean reconstruction of the
environment, which is required for the path planning and navigation of the planetary rover. Calibration done before a launch will become invalid because of vibration and temperature fluctuations during flight hence calibration needs to be done
on-site. Virtual objects are often used to enhance movies, and sometimes the whole
of the film is produced digitally. To generate the virtual object with the associated
virtual camera and to composite it with a real image sequence [3], the real camera
parameters must be known with better accuracy.''')
         

st.write("## Need of Calibration")         
st.write('''An accurately calibrated camera is needed for the above-mentioned applications to
work. Factory calibration (when the camera is delivered from the manufacturing
plant) is not valid because parameters change over the operating lifetime of the
device due to normal wear and tear, and thermal effects.
Possible causes of change in camera parameters over the lifetime of camera operation are:\n
(a) Thermal heat generated from the camera during its operation can cause the
focal length of the lens or CCD array expansion which doesn’t match with the
factory calibration (Fig. 1.3a),\n
(b) Because of mechanical stress the printed circuit board (PCB) attached to the 
camera module can bend which alters the camera pose see Fig. 1.3b,\n
(c) The non-rigid camera component can move and will change the calibration parameters as shown in Fig. 1.3c.\n
Camera calibration at the consumer end is not feasible due to the requirement
of buying accurate calibration patterns and thereafter collecting calibration data.
Hence dynamic camera calibration is an alternate economical and scalable way of
camera calibration.''')
st.image('Images/needDynCalib.jpg', 
         caption='''Figure 1.3: Causes of camera parameter offset (Image credit - [1])''',
                width=400)


st.write("## Classification Method")
st.write('''
Camera calibration techniques can be classified into two categories: photogrammetric 
calibration based on 3D objects as reference and self-calibration.\n
• Calibration based on three-dimensional reference objects: Camera calibration
is performed by viewing a calibration object whose shape in three dimensions
is well known. Typically, the calibration object is made up of two or three orthogonal planes. These methods need a complex setup and a costly calibration
apparatus. \n
• Self-calibration: This technique doesn’t involve the use of a calibration object.
Simply moving a camera in a static environment, imposes two limitations on
the internal camera parameters from a single camera displacement. Consequently, correspondences between the three photos are adequate to recover
both the internal and external parameters, allowing us to reconstruct the 3D
structure up to a similarity (when images are captured by the same camera
with fixed internal parameters).\n
We are investigating the factory calibration using the checkerboard pattern, however, it is also possible to calibrate using the orthogonal checkerboard pattern [4],
circular pattern [5], and the ArUco marker.
''')


st.write("## Problem Statement and Objectives")
st.write('''
Accurate calibration of camera parameters is crucial for several applications, including 3D measurements. In order to achieve this, both intrinsic and extrinsic
parameters must be calculated. Typically, calibration patterns are used for this purpose. However, in some cases, such as when the camera parameters change after
initial calibration or when using patterns is not feasible, alternative methods are
necessary. Dynamic and self-calibration provide a solution in such cases by capturing images of the 3D world and using feature matching to perform calibration with
multiple images. This project explores a solution for stereo camera calibration using
multiple images captured in a given scene, assuming that the intrinsic parameters
are already known.\n
• To calibrate a stereo camera by both factory and dynamic camera calibration.\n
• Estimation of stereo camera parameters (K, Rs and Ts).\n
• Verification of the camera parameter by depth estimation.
''')
