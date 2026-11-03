# Factory and Dynamic Stereo Camera Calibration

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://lakshay-bansal-webapp-dynamic-stereo-camera-calibra-home-untgaa.streamlit.app/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-Supported-ED6C00.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end framework and research toolkit for **Factory (checkerboard-based)** and **Dynamic (natural-scene / self-calibration)** camera calibration for monocular and stereo camera setups. Includes mathematical algorithms for non-linear bundle adjustment, multi-view triangulation, epipolar geometry, sample calibration datasets, and an interactive **Streamlit web application** for real-time 3D visualization.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Why Dynamic Calibration Matters](#why-dynamic-calibration-matters)
- [Repository Structure](#repository-structure)
- [Documentation & Deep Dives](#documentation--deep-dives)
- [Calibration Algorithms & Methods](#calibration-algorithms--methods)
  - [1. Factory Calibration](#1-factory-calibration-stereo-capture-multi-view)
  - [2. Dynamic Self-Calibration](#2-dynamic-self-calibration)
  - [3. Baseline OpenCV Implementations](#3-baseline-opencv-implementations)
- [Interactive Streamlit Web Application](#interactive-streamlit-web-application)
  - [Why the Webapp is Useful](#why-the-webapp-is-useful)
  - [Live Demo & Application Modules](#live-demo--application-modules)
  - [Running the Webapp Locally](#running-the-webapp-locally)
- [Datasets](#datasets)
- [Quick Start & Installation](#quick-start--installation)
- [References & Bibliography](#references--bibliography)
- [License & Author](#license--author)

---

## Project Overview

Multi-camera and stereo vision systems are essential for depth perception, 3D reconstruction, robotics navigation, autonomous driving, and computational photography (such as smartphone portrait mode and object dimension estimation). 

Accurate 3D geometry relies on knowing:
1. **Intrinsic parameters ($K$)**: Focal length, principal point offset, and lens distortion coefficients.
2. **Extrinsic parameters ($[R \mid T]$)**: Relative 3D rotation and translation between stereo cameras or camera poses across multiple views.

Traditional calibration relies on precise physical targets (e.g., checkerboard or ChArUco patterns). However, once a device leaves the manufacturing plant, environmental factors cause physical changes that invalidate factory parameters. This repository implements and evaluates:
- **Factory Calibration**: High-precision multi-view stereo calibration using planar checkerboard patterns and non-linear bundle adjustment.
- **Dynamic / Self-Calibration**: Online recalibration operating on arbitrary 3D outdoor and indoor scenes without artificial targets, using natural feature correspondences, essential matrix decomposition, and continuous parameter optimization.

---

## Why Dynamic Calibration Matters

Factory calibration is accurate only at the time of manufacturing. Over the device's operating lifetime, camera parameters undergo significant drift due to:

1. **Thermal Effects**: Heat generated during prolonged operation causes thermal expansion of the camera chassis, altering lens focal lengths and sensor plane distances.
2. **Mechanical Stress**: Dropping or flexing printed circuit boards (PCBs) shifts baseline distances ($T$) and relative camera orientations ($R$).
3. **Non-Rigid Structural Deformations**: Micro-vibrations in automotive, drone, or handheld setups induce dynamic distortions.

<p align="center">
  <img src="https://user-images.githubusercontent.com/84389082/212613466-abcb8bf3-4f70-40ad-aa8a-327a825c251d.jpg" width="45%" alt="Portrait Mode Depth Blur" />
  <img src="https://user-images.githubusercontent.com/84389082/212614651-b8a30d7f-806f-43e8-be54-a028f54bac56.png" width="45%" alt="3D Object Dimension Detection" />
</p>
<p align="center">
  <b>Fig. 1:</b> Applications requiring precise calibration: <i>(Left)</i> Depth-based portrait background blur; <i>(Right)</i> 3D object dimension and distance measurement.
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/84389082/212615261-6871efb1-c53d-4e7e-bf15-cf19436f6864.jpg" width="30%" alt="Thermal expansion" />
  <img src="https://user-images.githubusercontent.com/84389082/212615265-23d399bf-0f68-4ab3-a3de-74e8d2832679.jpg" width="30%" alt="Mechanical stress" />
  <img src="https://user-images.githubusercontent.com/84389082/212615267-c678d5a4-2e7b-4c23-be8c-6765225aa66d.jpg" width="30%" alt="Non-rigid deformation" />
</p>
<p align="center">
  <b>Fig. 2:</b> Causes of camera parameter drift: <b>(a)</b> Thermal expansion of lens & sensor array; <b>(b)</b> PCB bending from mechanical stress; <b>(c)</b> Non-rigid structural component movement. <i>(Source: Hagemann et al., WACV 2022)</i>
</p>

Because end users cannot be expected to carry checkerboard targets to recalibrate their devices, **dynamic calibration** uses natural scene imagery to continuously recover and refine camera parameters.

---

## Repository Structure

```text
Factory-and-Dynamic-Stereo-Camera-Calibration/
├── Code/
│   ├── A. Factory Calibration/
│   │   └── Stereo-Capture Multi-view Camera Calibration/
│   │       ├── main.py                     # Pipeline entry point for factory calibration
│   │       ├── bundle_adjustment.py        # Non-linear Levenberg-Marquardt bundle adjustment
│   │       ├── threshold_view_selection.py # Selection of optimal camera views based on coverage
│   │       ├── triangulation_nView.py      # N-view linear and non-linear 3D point triangulation
│   │       ├── utils.py                    # Projection, Rodrigues conversion, and plotting utilities
│   │       ├── archive.py                  # Auxiliary and legacy routines
│   │       └── SFM_CamerMatrix.txt         # Precomputed structure-from-motion camera matrices
│   │
│   ├── B. Dynamic Calibration/
│   │   ├── 1. Single Capture Dynamic Calibration Algorithm (SCDCA)/
│   │   │   ├── main.py                     # SCDCA execution script for monocular/dynamic scenes
│   │   │   ├── bundle_adjustment.py        # Least-squares bundle adjustment for SCDCA
│   │   │   ├── K_mat.txt                   # Reference intrinsic camera matrix
│   │   │   └── readme.md                   # Instructions for running SCDCA with ETH3D datasets
│   │   └── 2. Stereo-Capture Multi-view Dynamic Camera Calibration/
│   │       ├── main.py                     # Multi-view dynamic calibration pipeline
│   │       ├── bundle_adjustment.py        # Joint stereo bundle adjustment
│   │       ├── depth_hist.py               # Depth distribution histogram computation
│   │       └── useful_fn.py                # Feature matching, essential matrix, and pose helpers
│   │
│   └── Opencv Inbuilt Camera Calibration/
│       ├── Monocular Camera/
│       │   └── MATLAB/                     # Standard MATLAB camera calibration scripts (.m, .mlx, .pdf)
│       └── Stereo Camera Setup/
│           ├── MATLAB/                     # Stereo calibration scripts & live scripts (.m, .mlx)
│           └── Python/                     # Jupyter notebook and OpenCV calibration utilities
│
├── Dataset/
│   ├── Dynamic Scene/
│   │   ├── Monocular Camera/               # Outdoor dynamic sequences (Scene1, Scene2)
│   │   └── Stereo Setup/kusvod2/           # Standard Kusvod2 dynamic stereo benchmark dataset
│   ├── Static Scene/                       # Controlled static scene calibration frames
│   └── Zhang_Dataset/                      # Classical Zhang checkerboard dataset (left/right frames & results)
│
├── doc/
│   ├── 1_Camera Model.md                   # Pinhole model, radial distortion, and coordinate frames
│   ├── 2_Algorithms.md                     # Detailed algorithmic flowcharts and mathematical steps
│   └── Reference.md                        # Academic bibliography and literature citations
│
├── webapp/
│   ├── Home.py                             # Streamlit application main entry point
│   ├── requirements.txt                    # Python dependencies specific to the web application
│   ├── Readme.md                           # Web application documentation and live link
│   ├── LICENSE                             # Web application license
│   ├── pages/                              # Multi-page interactive Streamlit views
│   │   ├── 1_Introduction.py               # Project background and problem motivation
│   │   ├── 2_Literature Survey.py          # State-of-the-art review and literature summary
│   │   ├── 3_Camera Model.py               # Interactive camera models & lens distortion theory
│   │   ├── 4_Algorithms.py                 # Algorithmic breakdowns of SCDCA and multi-view calibration
│   │   ├── 5_Stereo-capture Multi-view Camera Factory Calibration.py # Interactive 3D bundle adjustment visualizer
│   │   ├── 6_Future Direction.py           # Ongoing improvements, deep learning calibration, edge deployment
│   │   └── 7_References.py                 # Interactive academic bibliography with direct paper links
│   ├── Dataset/                            # Embedded sample datasets for live browser testing
│   └── Images/                             # Diagnostic charts, diagrams, and UI figures
│
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation
```

---

## Documentation & Deep Dives

For mathematical formulations, step-by-step proofs, and implementation theory, consult the dedicated technical guides in the [`doc/`](doc) folder:

- **[Camera Model & Mathematical Foundations (`doc/1_Camera Model.md`)](doc/1_Camera%20Model.md)**:
  - Pinhole camera geometry and perspective projection.
  - Coordinate system transformations: World coordinates $\rightarrow$ Camera coordinates $\rightarrow$ Image plane coordinates.
  - Radial and tangential lens distortion modeling and correction equations.

- **[Algorithmic Architecture & Flowcharts (`doc/2_Algorithms.md`)](doc/2_Algorithms.md)**:
  - Detailed flowcharts and operational steps for Factory Calibration.
  - Mathematical steps of the Single Capture Dynamic Calibration Algorithm (SCDCA).
  - Multi-view candidate selection, essential matrix computation via the 5-point algorithm, RANSAC outlier rejection, and joint reprojection minimization.

- **[Academic Literature & References (`doc/Reference.md`)](doc/Reference.md)**:
  - Citations for seminal works in computer vision, including Zhang (2000), Hartley & Zisserman (Multiple View Geometry), Kumar et al. (CVPRW 2018), and Hagemann et al. (WACV 2022).

---

## Calibration Algorithms & Methods

### 1. Factory Calibration (Stereo-Capture Multi-View)
Factory calibration estimates high-accuracy initial intrinsic and extrinsic parameters using planar targets:
1. **Corner Detection**: Automatic sub-pixel corner extraction across stereo image pairs.
2. **Initial Pose Estimation**: Closed-form computation of individual camera extrinsic poses $[R_i \mid T_i]$ per view.
3. **N-View Triangulation**: Triangulation of matching 2D corner coordinates into initial 3D world coordinate estimates.
4. **Non-Linear Bundle Adjustment**: Simultaneous non-linear least-squares optimization minimizing the total reprojection error:
   $$\min_{R_s, T_s, R_i, T_i, P_j} \sum_{i} \sum_{j} \left\| x_{ij} - \pi(K, R_i, T_i, P_j) \right\|^2$$
   Parameters are parameterized using **Rodrigues rotation vectors** (3 parameters) and Euclidean 3D coordinates to ensure stable numerical convergence.

### 2. Dynamic Self-Calibration
Dynamic calibration eliminates the requirement for physical targets by processing natural scene images:
- **Single Capture Dynamic Calibration Algorithm (SCDCA)**:
  1. Dense feature extraction and guided matching between views.
  2. Relative pose estimation using the 5-point algorithm inside RANSAC.
  3. 3D point cloud triangulation from matched correspondences.
  4. Non-linear bundle adjustment refining extrinsic parameters and 3D structure.
- **Stereo-Capture Multi-View Dynamic Calibration**:
  Maintains a sliding candidate pool of multi-view frame sets ($I_i \in \{1, \dots, N\}$) against a reference capture ($I_{\text{ref}}$) to dynamically detect depth variance, verify epipolar constraints, and recalibrate parameters on-the-fly.

### 3. Baseline OpenCV Implementations
Standard calibration routines are provided under [`Code/Opencv Inbuilt Camera Calibration/`](Code/Opencv%20Inbuilt%20Camera%20Calibration/) in both **MATLAB** (`.m`, `.mlx`) and **Python** (Jupyter notebook), providing a ground-truth baseline against which the custom bundle adjustment pipelines are benchmarked.

---

## Interactive Streamlit Web Application

### Why the Webapp is Useful

Command-line calibration scripts typically print numerical matrices and cost values to a terminal. While useful for automation, this makes it difficult to understand:
- How 3D point clouds look in space before vs. after bundle adjustment.
- How reprojection errors are distributed across views.
- How changing camera parameters affects real-world depth estimation.

The **Streamlit Web Application** transforms complex calibration theory into an accessible, interactive visual platform:
- **Interactive 3D Point Cloud Inspection**: Rotate, pan, and zoom into reconstructed 3D points and camera frustums.
- **Visual Algorithm Walkthrough**: Step through each phase of factory and dynamic calibration with flowcharts and formula breakdowns.
- **Built-in Dataset Viewer**: Explore calibration patterns and dynamic scene images directly in the browser.
- **Educational & Research Showcase**: Ideal for students, researchers, and engineers exploring multi-view geometry.

### Live Demo & Application Modules

The application is deployed and publicly accessible:

> **Live Demo**: [https://lakshay-bansal-webapp-dynamic-stereo-camera-calibra-home-untgaa.streamlit.app/](https://lakshay-bansal-webapp-dynamic-stereo-camera-calibra-home-untgaa.streamlit.app/)

The webapp is structured into seven modular pages:
1. **About / Home**: High-level problem statement and taxonomy of calibration types.
2. **Introduction**: Real-world motivation (portrait mode, depth sensors) and causes of parameter drift.
3. **Literature Survey**: Chronological synthesis of camera calibration research.
4. **Camera Model**: Theory of pinhole projection, perspective transform, and lens distortion.
5. **Algorithms**: Comprehensive flowcharts of SCDCA and multi-view dynamic calibration.
6. **Factory Calibration Visualizer**: Live calculation of stereo camera calibration and interactive 3D bundle adjustment.
7. **Future Directions & References**: Deep learning calibration approaches and complete bibliography.

### Running the Webapp Locally

To run the interactive web application on your local machine:

```bash
# 1. Navigate to the webapp directory
cd webapp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run Home.py
```

Once started, open your browser at `http://localhost:8501`.

---

## Datasets

The repository includes curated datasets under [`Dataset/`](Dataset) for testing and benchmarking:

| Dataset | Type | Description |
| :--- | :--- | :--- |
| **Zhang Dataset** | Planar Checkerboard | Classical benchmark checkerboard images captured from left and right cameras, with ground-truth MATLAB calibration outputs (`Calib_Results_*.mat`). |
| **Kusvod2** | Dynamic Stereo Scene | Multi-view outdoor image pairs with diverse depth distributions and precomputed vanishing points / fundamental matrices. |
| **Scene1 & Scene2** | Monocular Dynamic | Continuous dynamic sequences captured across outdoor scenes for evaluating monocular dynamic self-calibration algorithms. |
| **Static Scene** | Controlled Indoor | Controlled static scenes for baseline epipolar geometry and feature-matching evaluations. |

---

## Quick Start & Installation

### Prerequisites
- Python 3.8 or higher
- MATLAB R2020b or later (optional, for running MATLAB `.mlx` / `.m` calibration scripts)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lakshay-Bansal/Factory-and-Dynamic-Stereo-Camera-Calibration.git
   cd Factory-and-Dynamic-Stereo-Camera-Calibration
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r webapp/requirements.txt
   ```
   *Core packages used: `numpy`, `scipy`, `opencv-python-headless`, `matplotlib`, `pandas`, `seaborn`, `streamlit`.*

### Running Factory Calibration
```bash
cd "Code/A. Factory Calibration/Stereo-Capture Multi-view Camera Calibration"
python main.py
```

### Running Dynamic Calibration (SCDCA)
```bash
cd "Code/B. Dynamic Calibration/1. Single Capture Dynamic Calibration Algorithm (SCDCA)"
python main.py
```

---

## References & Bibliography

Key academic literature underpinning this repository:
1. **Z. Zhang (2000)**: *"A flexible new technique for camera calibration"*, IEEE TPAMI.
2. **R. Hartley & A. Zisserman (2004)**: *"Multiple View Geometry in Computer Vision"*, Cambridge University Press.
3. **A. Kumar et al. (2018)**: *"Multicapture dynamic calibration of multi-camera systems"*, IEEE/CVF CVPRW.
4. **A. Hagemann, M. Knorr, and C. Stiller (2022)**: *"Modeling dynamic target deformation in camera calibration"*, IEEE/CVF WACV.
5. **C. Kurz, T. Thormählen, and H.-P. Seidel (2011)**: *"Bundle adjustment for stereoscopic 3D"*, Springer.

For the full list of 23 citations, see [`doc/Reference.md`](doc/Reference.md).

---

## License & Author

- **Author**: [Lakshay Bansal](https://github.com/Lakshay-Bansal)
- **License**: Distributed under the [MIT License](LICENSE). Feel free to use, modify, and distribute with proper attribution.
