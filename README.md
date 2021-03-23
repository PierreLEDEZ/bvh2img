# Bvh2Img

## Table of contents
  - [General info](#general-info)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Encoding techniques](#encoding-techniques)
    - [RGB](#rgb)
    - [Geometric features](#geometric-features)
      - [Example](#example)

## General info

This is a python script to convert BVH files from motion capture equipment ("Perception Neuron") to images with differents encoding processes.
The converted images are then used to train a CNN. 

## Installation

```console
# clone the repo
$ git clone URL REPO

# change the working directory to bvh2img
$ cd bvh2img

# install the requirements
$ pip install -r requirements.txt
```

## Usage
  
```console
# run the script
$ python main.py [--test] [--config path_to_config_file] [-h]
  optional arguments:
    -h, --help                                            Show this help message and exit
    -s, --save                                            Test the script on 10 files
    -c path_to_config_file, --config path_to_config_file  Use a config file

```

## Configuration
  
The config file shoud look like this.  
```ini
[SCRIPT]
encoding=geometric
coordinate=TR

[IMAGE]
width=236
height=118
scale_factor=1.05

[INTERPOLATION]
interpolation=True
method=bicubic

[ENHANCEMENT]
enhancement=True
clip_limit=10
```
## Encoding techniques

### RGB

The encoding process that I called "RGB" is simply converting the rotation and translation coordinates present in the BVH files into pixel values.  
For a given motion sequence $S$ of $N$ frames, noted $S=\{F1, F2, ..., Fn\}$, and a skeleton containing $M$ joints, each frame will contain $3*M$ translation coordinates and $3*M$ rotation coordinates.  
With the "Axis Neuron" software, the data are ordered as follows:
"Tx1 Ty1 Tz1 Ry1 Rx1 Rz1 Tx2 ... RyM RxM RzM"

* The rotation coordinates are normalized between 0 and 255 with:  
$R = 255 * \frac{(Rx - min(X_{rot}))}{(max(X_{rot})-min(X_{rot}))}$  
$G = 255 * \frac{(Ry - min(Y_{rot}))}{(max(Y_{rot})-min(Y_{rot}))}$  
$B = 255 * \frac{(Rz - min(Z_{rot}))}{(max(Z_{rot})-min(Z_{rot}))}$    
where $X_{rot}$, $Y_{rot}$ and $Z_{rot}$ represent respectively rotation coordinates around X, Y and Z axis of all frames.

* The translation coordinates are normalized with the same formula, except that the coordinates is multiplied by a scale factor  
$R = 255 * \frac{(Tx*Scale - min(X_{tra}))}{(max(X_{tra})-min(X_{tra}))}$  
$G = 255 * \frac{(Ty*Scale - min(Y_{tra}))}{(max(Y_{tra})-min(Y_{tra}))}$  
$B = 255 * \frac{(Tz*Scale - min(Z_{tra}))}{(max(Z_{tra})-min(Z_{tra}))}$  
where $X_{tra}$, $Y_{tra}$ and $Z_{tra}$ represents respectively translation coordinates along X, Y and Z axis of all frames.

The resulting image is structured as follows: 
A frame is represented by a column in the image (one pixel wide and the height is equal to the number of joints multiplied by two).  
Each column is organized in the same way, from top to bottom there are two pixels per joint.  
The first pixel of a joint includes the 3 color values (R, G and B) previously calculated corresponding to the translation coordinates. The second pixel consists of the 3 values (R,G and B) calculated from the coordinates of rotation.  
The order of the joints in a column follows the hierarchy described in the BVH file (head, legs, trunk and arms).  
Finally, an image contains as many columns as there are frames. 

**Example image**  

### Geometric features

This encoding process is similar to the one presented by Huy-Hieu Pham in his work (citation here), the resulting images are called "SPMF" (Skeleton Pose-Motion Feature) or "Enhanced-SPMF".  

From the XYZ coordinates of joints, two elements has to be extracted: PF (Pose Feature) and MF (Motion Feature)  

* PF : it's composed of two geometric features, the Joint-Joint Distance ($JJD$) and the Joint-Joint Orientation ($JJO$).  
  * The $JJD$ feature is the euclidean distance between two joints in a frame. This is computed for each frame and for every joints in a frame (To avoid useless computation time, the distance from a joint (A) to another (B) is computed but the distance between B and A is ignored because it's the same)  
  Once every distance is computed, they are normalized between $[0;255]$ and next, a colormap is applied with OpenCV to convert the scalar distance to a pixel with 3 values (R, G, and B) 
  * The $JJO$ feature is the orientation vector between two joints in a given frame. As the $JJD$, it's computed for every joints in each frame and for two given joints, just one orientation is computed.  
  * For a given frame $t$, $PF^t=[JJD^t ++ JJO^t]$ (here "++" represents concatenation between the two arrays). In term of image, $PF^t$ is a column, the upper half represents all the distances and the lower half all the orientations.
* MF : it's composed of the same features than PF. Here the $JJD$ and $JJO$ are computed between every joints in the frame $t$ and every joints in the frame $t+1$.  
  For two given frames $t-$ and $t+1$, $MF^{t->t+1}=[JJD^{t,t+1} ++ JJO^{t,t+1}]$. In term of image, $MF^{t->t+1}$ is a column where the upper half represents all the distances and the lower half all the orientations.  
  
The image is composed from PF and MF as follows:
$SPMF=[PF^1 ++ MF^{1->2} ++ PF^2 ++ ... ++ MF^{N-1->N} ++ PF^N]$ where N is the number of frames.  

Since the joint data received by the "Perception Neuron" mocap equipment are not all expressed in the same coordinate system, it is necessary to move them in the global coordinate system, the root one.  

A bvh file is separated in two main parts: Hierarchy and Motion. The hierarchy part describes the skeleton and his joints and the motion part describes the motion of each joint.    
Except for the Hips which is considered as the root joint, each joint has a parent and can have children.
In the hierarchy, each joint has an offset that represents the distance to its parent.  
In the motion part, for a given joint, coordinates (Tx, Ty, Tz, Ry, Rx and Rz) are expressed in the coordinate system of its parent.  

The global coordinate system is the one in which the root node is expressed. To obtain global coordinates for all joints, you need to compute the translation and rotation for each parent joint.  
  
#### Example
  
For this example, let's say the **root** coordinates are :  
$Tx_{root}=0.000000\ ;\ Ty_{root}=93.019646\ ;\ Tz_{root}=0.000000\ ;$  
$Rx_{root}=0.000000\ ;\ Ry_{root}=101.470901\ ;\ Rz_{root}=0.000000\ ;$    
  
The next ones in the bvh file correspond to the **upper left leg** and they are :  
$Tx_{ull}=-9.250000\ ;\ Ty_{ull}=-1.589645\ ;\ Tz_{ull}=0.000000\ ;$  
$Rx_{ull}=0.000000\ ;\ Ry_{ull}=0.000000\ ;\ Rz_{ull}=0.000000\ ;$  
  
The next ones in the bvh file correspond to the **left knee** and they are :  
$Tx_{lk}=0.000000\ ;\ Ty_{lk}=-41.870003\ ;\ Tz_{lk}=0.000000\ ;$  
$Rx_{lk}=0.000000\ ;\ Ry_{lk}=0.000000\ ;\ Rz_{lk}=0.000000\ ;$
    
This coordinates are relative to the direct parent. To express the upper left leg coordinates in the global coordinate system, you have to compute translation and rotation matrices.  

<!-- $\ce{^{R_G}_{}P_{ull}} = \begin{bmatrix}
x_{ull} \\
y_{ull} \\
z_{ull} \\
1
\end{bmatrix} = \ce{^{R_G}_{R_{root}}T} * \ce{^{R_G}_{R_{root}}R} * \ce{^{R_root}_{}P_{ull}}$ --> <img style="transform: translateY(0.1em); background: white;" src=".\svg\lk1SvMaPsD.svg">
  
where <!-- $
\ce{^{R_G}_{R_{root}}T} = 
\begin{bmatrix}
1 & 0 & 0 & Tx_{root} \\
0 & 1 & 0 & Ty_{root} \\
0 & 0 & 1 & Tz_{root} \\
0 & 0 & 0 & 1
\end{bmatrix}$ --> <img style="transform: translateY(0.1em); background: white;" src=".\svg\BqQLeEUkIX.svg"> and  
  
<!-- $\ce{^{R_G}_{R_{root}}R} =  \begin{bmatrix}cos(Ry_{root}) & 0 & sin(Ry_{root}) & 0 \\0 & 1 & 0 & 0 \\-sin(Ry_{root}) & 0 & cos(Ry_{root}) & 0 \\0 & 0 & 0 & 1\end{bmatrix} * \begin{bmatrix}1 & 0 & 0 & 0 \\0 & cos(Rx_{root}) & -sin(Rx_{root}) & 0 \\0 & sin(Rx_{root}) & cos(Rx_{root}) & 0 \\0 & 0 & 0 & 1\end{bmatrix} * \begin{bmatrix}cos(Rz_{root}) & -sin(Rz_{root}) & 0 & 0 \\sin(Rz_{root}) & cos(Rz_{root}) & 0 & 0 \\0 & 0 & 1 & 0 \\0 & 0 & 0 & 1\end{bmatrix}$ --> <img style="transform: translateY(0.1em); background: white;" src=".\svg\OG1UPzGvCH.svg">

Once this is done, the result is the coordinates of the upper left leg in the global coordinate system.  
Let's continue with the left knee, you can note $\ce{^{R_G}_{R_{root}}H} = \ce{^{R_G}_{R_{root}}T} * \ce{^{R_G}_{R_{root}}R}$.  
The global coordinates of the left knee are:  
  
$\ce{^{R_G}_{}P_{lk}} = \begin{bmatrix}
x_{lk} \\
y_{lk} \\
z_{lk} \\
1
\end{bmatrix} = \ce{^{R_G}_{R_{root}}H} * \ce{^{R_root}_{R_{ull}}H} * \ce{^{R_ull}_{}P_{lk}}$ with $\ce{^{R_root}_{R_{ull}}H}=\ce{^{R_root}_{R_{ull}}T} * \ce{^{R_root}_{R_{ull}}R}$, as calculated above.

