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


st.title("Camera Model")

st.write('''To familiarise with the process of calibration start calibrating a
monocular and stereo camera by using a conventional Tsai algorithm [13].''')

st.image('Images/PinHole_camera.jpg', 
         caption='''Figure 3.1: Pin hole camera model. 
         Source: https://in.mathworks.com/help/vision/ug/camera-calibration.html ''',
                width=300)

st.write('''
The pinhole camera is as shown in Fig. 3.1. The light reflected from an object that
needs to be captured passes through a pinhole and the image is formed on the
screen of the camera (inverted w.r.t. orientation of the object in the real world i.e.
captured by camera). In the pin-hole camera model, it is assumed that all light rays
are passed through the pinhole’s center before being imaged on the screen. Ideally,
pinhole camera model, which lacks a lens, does not consider lens distortion. The full
camera model that accounts for lens distortion includes radial and tangential lens
distortion.
''')

st.write('''
As shown in Fig. 3.2, the point P (Xw, Yw, Zw) in world coordinate is transformed
to (Xc, Yc, Zc) from the reference frame of camera coordinate system by applying
rotation and translation to the world point (see Eq. 1).
''')

st.image('Images/CameraModel_1.jpg', 
         caption='''Figure 3.2: Image formation of pin hole camera via a perspective transform ''',
                width=500)

st.latex(r'''
\begin{equation}
    \begin{bmatrix}
     X_c \\
     Y_c \\
     Z_c
    \end{bmatrix}
  = \mathbf{R}
 \begin{bmatrix}
     X_{w}\\
     Y_{w}\\ 
     Z_{w}
 \end{bmatrix}
 +
\mathbf{T},
\end{equation}
''')
st.write('''where  $\\mathbf{R}$ is a $3 \\times 3$ rotation matrix and  $\\mathbf{T}$ is a $3 \\times 1$ translation matrix. 
         Then (Xc, Yc, Zc) is mapped to the image plane after perspective projection.''')

st.latex(r'''
\begin{equation}
x = f\frac{X_{c}}{Z_{c}} \quad\text{,}\quad y = f\frac{Y_{c}}{Z_{c}}
\end{equation}
''')
st.write("Which can be written as:")
st.latex(r'''
\begin{equation}
     \begin{bmatrix}
        x_{im}\\ 
        y_{im}\\
        z_{im}
    \end{bmatrix} = \mathbf{K}
    \begin{bmatrix}
        X_{c}\\
        Y_{c}\\
        Z_c
    \end{bmatrix},
\end{equation} ''')
st.write("where $\\mathbf{K}$ is intrinsic matrix and sub-script 'im' signifies image plane.")

st.latex(r'''
\begin{equation}
    \mathbf{K} = 
    \begin{bmatrix}
        \alpha f & 0 & o_x \\
        0 & \beta f & o_y \\
        0 & 0 & 1
    \end{bmatrix},
\end{equation}
''')
st.write('''where $\\alpha$ and $\\beta$ are scale factor. For square pixels  $\\alpha$ and $\\beta$ are equal to 1.
         Hence (Xc, Yc, Zc) mapped to the ($x_{im}, y_{im}, z_{im}$) in image plane in homogeneous coordinate form by Eq. 3 
         as shown in expanded form below.''')

st.latex(r'''
\begin{equation}
   \begin{bmatrix}
        x_{im}\\ 
        y_{im}\\
        z_{im}
    \end{bmatrix}_{\substack{\text{homogeneous} \\ \text{coord.}}}
    =
    \begin{bmatrix}
        \alpha f & 0 & o_x \\
        0 & \beta f & o_y \\
        0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        X_{c}\\
        Y_{c}\\
        Z_c
    \end{bmatrix}
    = 
    \begin{bmatrix}
        \alpha f X_c + o_x Z_c \\
        \beta f Y_c + o_y Z_c \\
        Z_c
    \end{bmatrix}
\end{equation}
''')

st.write("x and y in Eq. 2 can be obatined from Eq. 5 as:")
st.latex(r'''
\begin{equation}
x = \frac{x_{im}}{z_{im}} \quad\text{,}\quad y = \frac{y_{im}}{z_{im}}
\end{equation}
''')

