#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:16:27 2023

@author: lakshayb
"""

import streamlit as st
import matplotlib.pyplot as plt
import os
import sys

PWD = os.getcwd()
sys.path.append(rf'{PWD}\Images')

st.title("About")

st.write('''
Project explore the estimation of the camera parameters such as intrinsic (focal length, camera offset) and 
extrinsic parameters (rotation and translation of stereo setup) of the stereo camera.
''')

st.text("We can classify camera calibration in two broad types:")
st.text("1. Factory Calibration")
st.text("2. Dynamic Calibration")

st.image('Images/flow_calib.jpg', 
         caption='''Figure : Types of camera calibration.''',
                width=400)

st.write('''
1. Factory calibration - Camera parameters are estimated by using checkerboard pattern 
so as to make 3D measurement (depth) more accurate by performing bundle adjustment.
2. Dynamic calibration - Calibration using real scene images for correcting the possible deformation due to
transportation and handling.
''')
