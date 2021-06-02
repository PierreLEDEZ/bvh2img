import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse
import math
import os
import random
import threading
from pathlib import Path

import cv2
import numpy as np

import lib.bvh2geometric as bvh2geometric
from src.bvh import bvh_parser
from src.constants import *
from src.utils import *

# Directory which contains the BVH files
DATADIR = "../data/"

# For windows terminal, activate colors
if os.name == "nt":
    os.system("color")

#Names of joints we want to ignore (Fingers of both hands)
IGNORED_JOINTS = [
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightInHandIndex",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightInHandMiddle",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightInHandRing",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightInHandPinky",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftInHandIndex",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftInHandMiddle",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftInHandRing",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftInHandPinky",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3"
]

def plotFrame(frame):
    """
        Plot all joints of one frame in 3D

        :param frame: Position of the body (frame must contain global coordinates)
        :type frame: numpy array 
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    # Separate the 3 axis into 3 distincts arrays
    x = []
    y = []
    z = []
    for elt in frame:
        if elt.all() == None:
            continue
        x.append(elt[0])
        y.append(elt[1])
        z.append(elt[2])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x, y, z, zdir="z")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_zlim(-180, 180)
    plt.show()

def getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index):
    """
        Calculate world coordinates of the body joints for each frame

        :param bvhParser: the parser used to parse the BVH file
        :type bvhParser: BVHParser Class
        :param joints: The list of joints in the body
        :type joints: array of Joints Object
        :param ignored_joints: The list of joints we want to ignore
        :type ignored_joints: array of string

        :return: The world coordinates of the joints for all parsed frames
        :rtype: numpy array

    """

    # image's width = number of frames, image's height = number of joints - number of ignored joints
    width, height = len(frames), (len(joints)-len(ignored_joints))

    coords_frames = np.zeros((width, height, 3), dtype=np.float32)

    initial_Yrotation = 0

    for frame_index, frame in enumerate(frames):
        stack_joints = []
        stack_M = []
        joint_placed = 0
        pixel_placed = 0
        for joint_index in range(0, len(joints)):

            # if the index of the joint is in the list of ignored joints
            # put empty values in the 2 stacks and update the joint cursor (joint_placed)
            if joint_index in ignored_joints_index:
                stack_joints.append(np.array([None, None, None, None, None]))
                stack_M.append(np.eye(4))
                joint_placed += 6
                continue

            # first joint is the root -> Hips
            if joint_index == 0:
                # get current joint coordinates from bvh
                bvh_content = [frame[joint_placed + i] for i in range(6)]

                if frame_index == 0:
                    initial_Yrotation = bvh_content[3]
                    if initial_Yrotation < 0:
                        initial_Yrotation = 360 - np.abs(initial_Yrotation)
                    bvh_content[3] = 0.0

                else:
                    tmp = bvh_content[3]
                    if tmp < 0:
                        tmp = 360 - np.abs(tmp)
                    Yrotation = tmp - initial_Yrotation
                    bvh_content[3] = Yrotation
                # save bvh informations in the stack_joints array
                stack_joints.append(bvh_content)

                # At the beginning, matrix M is just the identity matrix
                stack_M.append(np.eye(4))
                
                # Put the first three coordinates in coordinates array because they are express in the global coordinate system
                coords_frames[frame_index][pixel_placed] = bvh_content[:3]
            
            else:
                # get current joint coordinates from bvh
                bvh_content = [frame[joint_placed+i] for i in range(6)]

                # get the matrix M of its parent
                M_parent = stack_M[joints[joint_index].parent.index]

                # retrieve informations about its parent to calculate the change of coordinate system
                Tx, Ty, Tz, Ry, Rx, Rz = stack_joints[joints[joint_index].parent.index]
                T = getTranslationMat(Tx, Ty, Tz)

                rot_mat_Y, rot_mat_X, rot_mat_Z = getRotationMatY(Ry), getRotationMatX(Rx), getRotationMatZ(Rz)
                R = rot_mat_Y.dot(rot_mat_X).dot(rot_mat_Z)

                # compute new matrix M with current rotation and translation matrices
                M = T.dot(R)
                M = M_parent.dot(M)
                
                # express the local coordinates in the homogeneous coordinates (add a 1 to the fourth dimensions)
                local_coordinates = np.concatenate([bvh_content[:3], np.array([1])])

                # compute global coordinates of this joint and add them to the coordinates array
                global_coordinates = M.dot(local_coordinates)
                coords_frames[frame_index][pixel_placed] = global_coordinates[:3]
                
                # save current matrix M in stack_M and local informations of this joint in stack_joints
                stack_M.append(M)
                stack_joints.append(bvh_content)
            
            joint_placed += 6
            pixel_placed += 1

    with open("class_example.npy", "wb") as f:
        np.save(f, coords_frames)

    return coords_frames

def bvh2LiuImg(frames, joints, ignored_joints, ignored_joints_index):
    # First, compute global coordinates from bvh informations
    coords_frames = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)

    F = len(coords_frames)
    N = len(coords_frames[0])

    space_2D = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4]
        ],
        dtype=np.uint8
    )

    space_3D = np.array(
        [
            [2, 3, 4],
            [1, 3, 4],
            [1, 2, 4],
            [1, 2, 3],
            [0, 3, 4],
            [0, 2, 4],
            [0, 2, 3],
            [0, 1, 4],
            [0, 1, 3],
            [0, 1, 2],
        ],
        dtype=np.uint8
    )

    energies = np.zeros((N), dtype=np.float32)
    for f in range(F-1):
        current_frame = coords_frames[f]
        next_frame = coords_frames[f+1]
        for n in range(N):
            energy = np.linalg.norm(next_frame[n] - current_frame[n])
            energies[n] += energy

    energies = (energies - np.amin(energies))/(np.amax(energies) - np.amin(energies))

    rho = .8

    weights = rho*energies + (1 - rho)

    images = [[[], [] ,[]]]

    for f in range(F):
        for n in range(N):
            coords_5D = np.concatenate((coords_frames[f][n], np.array([f, n])))
            for c in range(10):
                current_2D = space_2D[1]
                current_3D = space_3D[1]
                j = coords_5D[current_2D[0]]
                k = coords_5D[current_2D[1]]
                r = coords_5D[current_3D[0]]
                g = coords_5D[current_3D[1]]
                b = coords_5D[current_3D[2]]
                rgb = np.array([r, g, b])
                rgb = (1 - weights[n])*np.array([255, 255, 255]) + weights[n]*rgb
                images[0][0].append(j)
                images[0][1].append(k)
                images[0][2].append(rgb)

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.axis("off")

    y = images[0][0]
    x = images[0][1]

    rgb = np.array(images[0][2])
    rgb = (rgb - np.amin(rgb))/(np.amax(rgb) - np.amin(rgb))
    
    ax.scatter(x, y, s=50, facecolors=rgb)
    fig.canvas.draw()
    X = np.array(canvas.renderer.buffer_rgba())
    X = cv2.cvtColor(X, cv2.COLOR_RGB2BGR)

    return X

def bvh2GeometricFeaturesPham(frames, joints, ignored_joints, ignored_joints_index):
    """
        Convert BVH informations to RGB image after computing geometric features

        :param bvhParser: parser used to parse the given bvh
        :type bvhParser: BVHParser Class
        :param joints: List of body joints
        :type joints: array of Joint Object
        :param ignored_joints: List of joints we want to ignore
        :type ignored_joints: array of string

        :return: RGB image
        :rtype: numpy array

        .. note::
            This function is also in the lib folder in bvh2geometric.pyx file "converted" in Cython
    """

    # First, compute global coordinates from bvh informations
    coords_frames = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)

    # Arrays to store distances and orientations for pose features (PF) and motion features (MF)
    D_original_pose = []
    O_original_pose = []
    D_original_motion = []
    O_original_motion = []

    # For each frame, compute all JJD (Joint-Joint Distance) and JJO (Joint-Joint Orientation) for Pose Features (PF) and Motion Features (MF)
    for t in range(len(coords_frames)):
        frame_t = coords_frames[t]

        # JJD and JJO in MF are calculated with 2 frames
        if t != len(coords_frames)-1:
            next_frame = coords_frames[t+1]

        JJD_pose = []
        JJD_motion = []
        JJO_pose = []
        JJO_motion = []

        # for each joint J, calculate JJD and JJO with all of the other joints
        for joint_J in range(len(frame_t)-1):
            for joint_K in range(joint_J+1, len(frame_t)):
                euclidean_distance_pose = math.sqrt( 
                    (frame_t[joint_J][0] - frame_t[joint_K][0])**2 + 
                    (frame_t[joint_J][1] - frame_t[joint_K][1])**2 + 
                    (frame_t[joint_J][2] - frame_t[joint_K][2])**2
                )
                JJD_pose.append(euclidean_distance_pose)
                
                if t != len(coords_frames)-1:
                    euclidean_distance_motion = math.sqrt( 
                        (frame_t[joint_J][0] - next_frame[joint_K][0])**2 + 
                        (frame_t[joint_J][1] - next_frame[joint_K][1])**2 + 
                        (frame_t[joint_J][2] - next_frame[joint_K][2])**2 
                    )
                    JJD_motion.append(euclidean_distance_motion)
                
                orientation_pose = np.array(
                    [
                        frame_t[joint_K][0] - frame_t[joint_J][0],
                        frame_t[joint_K][1] - frame_t[joint_J][1],
                        frame_t[joint_K][2] - frame_t[joint_J][2]
                    ]
                )
                JJO_pose.append(orientation_pose)

                if t != len(coords_frames)-1:
                    orientation_motion = np.array(
                        [
                            frame_t[joint_K][0] - next_frame[joint_J][0],
                            frame_t[joint_K][1] - next_frame[joint_J][1],
                            frame_t[joint_K][2] - next_frame[joint_J][2]
                        ]
                    )
                    JJO_motion.append(orientation_motion)

        D_original_pose.append(JJD_pose)
        O_original_pose.append(JJO_pose)
        if t != len(coords_frames)-1: 
            D_original_motion.append(JJD_motion)
            O_original_motion.append(JJO_motion)

    D_original_pose = np.array(D_original_pose)
    D_original_motion = np.array(D_original_motion)

    # Normalize D_original_pose and D_original_motion between [0; 255] and apply JET color map on them to obtain two images with 3 channels (R, G, B)
    D_min_pose = 0
    D_min_motion = 0
    
    D_max_pose = np.amax(D_original_pose)
    D_max_motion = np.amax(D_original_motion)

    D_norm_pose = 255 * ((D_original_pose - D_min_pose) / (D_max_pose - D_min_pose))
    D_norm_pose = np.array(D_norm_pose, dtype=np.uint8)
    
    D_norm_motion = 255 * ((D_original_motion - D_min_motion) / (D_max_motion - D_min_motion))
    D_norm_motion = np.array(D_norm_motion, dtype=np.uint8)

    JJD_RGB_pose = cv2.applyColorMap(D_norm_pose, cv2.COLORMAP_JET)
    JJD_RGB_motion = cv2.applyColorMap(D_norm_motion, cv2.COLORMAP_JET)

    O_original_pose = np.array(O_original_pose)
    O_original_motion = np.array(O_original_motion)

    # Normalize between [0; 255] and consider (x, y, z) as (R, G, B)
    c_max_pose = np.amax(O_original_pose)
    c_max_motion = np.amax(O_original_motion)

    c_min_pose = np.amin(O_original_pose)
    c_min_motion = np.amin(O_original_motion)
    
    JJO_RGB_pose = np.array(np.floor(255 * ((O_original_pose - c_min_pose) / (c_max_pose - c_min_pose))), dtype=np.uint8)
    JJO_RGB_motion = np.array(np.floor(255 * ((O_original_motion - c_min_motion) / (c_max_motion - c_min_motion))), dtype=np.uint8)

    PF = np.zeros((len(coords_frames), 420,  3), dtype=np.uint8)
    MF = np.zeros((len(coords_frames)-1, 420, 3), dtype=np.uint8)

    for nb_frame in range(len(coords_frames)):
        temp_pf = np.concatenate([JJD_RGB_pose[nb_frame], JJO_RGB_pose[nb_frame]])
        PF[nb_frame] = temp_pf

    for nb_frame in range(len(coords_frames)-1):
        temp_mf = np.concatenate([JJD_RGB_motion[nb_frame], JJO_RGB_motion[nb_frame]])
        MF[nb_frame] = temp_mf

    PF = np.array(PF)
    MF = np.array(MF)
    SPMF = np.zeros((420, 2*len(coords_frames)-1, 3), dtype=np.uint8)

    frame_index = 0
    for pf, mf in zip(PF, MF):
        SPMF[:, frame_index] = pf
        frame_index += 1
        SPMF[:, frame_index] = mf
        frame_index += 1
    
    # Add the last element of PF to SPMF because the zip function keeps only the elements in common between the two tables. 
    # The PF table has one element more than MF.
    SPMF[:, frame_index] = PF[-1]
    SPMF = convertRGB2BGR(SPMF)

    return SPMF

def bvh2GeometricFeaturesRoot(frames, joints, ignored_joints, ignored_joints_index):
    """
        Convert BVH informations to RGB image after computing geometric features

        :param frames: frames contained in the bvh parsed file
        :type frames: numpy array
        :param joints: List of body joints
        :type joints: array of Joint Object
        :param ignored_joints: List of joints we want to ignore
        :type ignored_joints: array of string

        :return: RGB image
        :rtype: numpy array
    """

    coords_frames = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)

    # For each frame, compute all JJD (Joint-Joint Distance) and JJO (Joint-Joint Orientation) for Pose Features (PF) and Motion Features (MF)
    D_original_pose = []
    O_original_pose = []
    D_original_motion = []
    O_original_motion = []

    for t in range(len(coords_frames)):
        frame_t = coords_frames[t]
        if t != len(coords_frames)-1:
            next_frame = coords_frames[t+1]

        JJD_pose = []
        JJD_motion = []
        JJO_pose = []
        JJO_motion = []
        root_t = frame_t[0]

        for joint in range(1, len(frame_t)):
            euclidean_distance_pose = math.sqrt( 
                (frame_t[joint][0] - root_t[0])**2 + 
                (frame_t[joint][1] - root_t[1])**2 + 
                (frame_t[joint][2] - root_t[2])**2
            )
            JJD_pose.append(euclidean_distance_pose)
            
            if t != len(coords_frames)-1:
                euclidean_distance_motion = math.sqrt( 
                    (next_frame[joint][0] - root_t[0])**2 + 
                    (next_frame[joint][1] - root_t[1])**2 + 
                    (next_frame[joint][2] - root_t[2])**2 
                )
                JJD_motion.append(euclidean_distance_motion)
            
            orientation_pose = np.array(
                [
                    -frame_t[joint][0] + root_t[0],
                    -frame_t[joint][1] + root_t[1],
                    -frame_t[joint][2] + root_t[2],
                ]
            )
            JJO_pose.append(orientation_pose)
            if t != len(coords_frames)-1:
                orientation_motion = np.array(
                    [
                        -next_frame[joint][0] + root_t[0],
                        -next_frame[joint][1] + root_t[1],
                        -next_frame[joint][2] + root_t[2],
                    ]
                )
                JJO_motion.append(orientation_motion)

        D_original_pose.append(JJD_pose)
        O_original_pose.append(JJO_pose)

        if t != len(coords_frames)-1: 
            D_original_motion.append(JJD_motion)
            O_original_motion.append(JJO_motion)
    
    D_original_pose = np.array(D_original_pose)
    D_original_motion = np.array(D_original_motion)
    
    # Normalize D_original_pose and D_original_motion between [0; 255] and apply JET color map on them to obtain two images with 3 channels (R, G, B)
    D_min_pose = 0
    D_min_motion = 0
    
    D_max_pose = np.amax(D_original_pose)
    D_max_motion = np.amax(D_original_motion)

    D_norm_pose = 255 * ((D_original_pose - D_min_pose) / (D_max_pose - D_min_pose))
    D_norm_pose = np.array(D_norm_pose, dtype=np.uint8)
    
    D_norm_motion = 255 * ((D_original_motion - D_min_motion) / (D_max_motion - D_min_motion))
    D_norm_motion = np.array(D_norm_motion, dtype=np.uint8)

    JJD_RGB_pose = cv2.applyColorMap(D_norm_pose, cv2.COLORMAP_JET)
    JJD_RGB_motion = cv2.applyColorMap(D_norm_motion, cv2.COLORMAP_JET)

    O_original_pose = np.array(O_original_pose)
    O_original_motion = np.array(O_original_motion)

    # Normalize between [0; 255] and consider (x, y, z) as (R, G, B)
    c_max_pose = np.amax(O_original_pose)
    c_max_motion = np.amax(O_original_motion)

    c_min_pose = np.amin(O_original_pose)
    c_min_motion = np.amin(O_original_motion)
    
    JJO_RGB_pose = np.array(np.floor(255 * ((O_original_pose - c_min_pose) / (c_max_pose - c_min_pose))), dtype=np.uint8)
    JJO_RGB_motion = np.array(np.floor(255 * ((O_original_motion - c_min_motion) / (c_max_motion - c_min_motion))), dtype=np.uint8)

    PF = np.zeros((len(coords_frames), 40,  3), dtype=np.uint8)
    MF = np.zeros((len(coords_frames)-1, 40, 3), dtype=np.uint8)

    for nb_frame in range(len(coords_frames)):
        temp_pf = np.concatenate([JJD_RGB_pose[nb_frame], JJO_RGB_pose[nb_frame]])
        # temp_pf = JJD_RGB_pose[nb_frame]
        PF[nb_frame] = temp_pf
        if nb_frame < len(coords_frames)-1:
            temp_mf = np.concatenate([JJD_RGB_motion[nb_frame], JJO_RGB_motion[nb_frame]])
            # temp_mf = JJD_RGB_motion[nb_frame]
            MF[nb_frame] = temp_mf

    PF = np.array(PF)
    MF = np.array(MF)
    SPMF = np.zeros((40, 2*len(coords_frames)-1, 3), dtype=np.uint8)

    frame_index = 0
    for pf, mf in zip(PF, MF):
        SPMF[:, frame_index] = pf
        frame_index += 1
        SPMF[:, frame_index] = mf
        frame_index += 1
    
    # Add the last element of PF to SPMF because the zip function keeps only the elements in common between the two tables. 
    # The PF table has one element more than MF.
    SPMF[:, frame_index] = PF[-1]

    return SPMF

def calculate_euclidean_distance(joint_from, joint_to):
    return math.sqrt(
        (joint_to[0] - joint_from[0])**2 + 
        (joint_to[1] - joint_from[1])**2 +
        (joint_to[2] - joint_from[2])**2
    )

def calculate_orientation(joint_from, joint_to):
    return np.array(
        [
            joint_from[0] - joint_to[0],
            joint_from[1] - joint_to[1],
            joint_from[2] - joint_to[2],
        ]
    )

def bvh2GeometricFeaturesCustom(frames, joints, ignored_joints, ignored_joints_index, joints_we_want):
    """
        Convert BVH informations to RGB image after computing geometric features

        :param bvhParser: parser used to parse the given bvh
        :type bvhParser: BVHParser Class
        :param joints: List of body joints
        :type joints: array of Joint Object
        :param ignored_joints: List of joints we want to ignore
        :type ignored_joints: array of string

        :return: RGB image
        :rtype: numpy array
    """
    # joints_we_want = [0, 13, 16, 17, 20] # root, shoulders, hands
    # joints_we_want = [4, 7, 12, 16, 20] # head, feet, hands
    coords_frames = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)

    # For each frame, compute all JJD (Joint-Joint Distance) and JJO (Joint-Joint Orientation) for Pose Features (PF) and Motion Features (MF)
    D_original_pose = []
    O_original_pose = []
    D_original_motion = []
    O_original_motion = []

    for t in range(len(coords_frames)):
        if t == 0:
            plotFrame(coords_frames[t])
        frame_t = coords_frames[t]
        if t != len(coords_frames)-1:
            next_frame = coords_frames[t+1]

        JJD_pose = []
        JJD_motion = []
        JJO_pose = []
        JJO_motion = []

        references = [frame_t[j] for j in joints_we_want]

        for joint_J in range(len(frame_t)):
            for joint_K, ref in zip(joints_we_want, references):
                if joint_J == joint_K:
                    continue

                JJD_pose.append(calculate_euclidean_distance(ref, frame_t[joint_J]))
                
                if t != len(coords_frames)-1:
                    JJD_motion.append(calculate_euclidean_distance(ref, next_frame[joint_J]))
                
                JJO_pose.append(calculate_orientation(ref, frame_t[joint_J]))

                if t != len(coords_frames)-1:
                    JJO_motion.append(calculate_orientation(ref, next_frame[joint_J]))

        D_original_pose.append(JJD_pose)
        O_original_pose.append(JJO_pose)

        if t != len(coords_frames)-1: 
            D_original_motion.append(JJD_motion)
            O_original_motion.append(JJO_motion)
    
    D_original_pose = np.array(D_original_pose)
    D_original_motion = np.array(D_original_motion)

    # Normalize D_original_pose and D_original_motion between [0; 255] and apply JET color map on them to obtain two images with 3 channels (R, G, B)
    D_min_pose = 0
    D_min_motion = 0
    
    D_max_pose = np.amax(D_original_pose)
    D_max_motion = np.amax(D_original_motion)

    D_norm_pose = 255 * ((D_original_pose - D_min_pose) / (D_max_pose - D_min_pose))
    D_norm_pose = np.array(D_norm_pose, dtype=np.uint8)
    
    D_norm_motion = 255 * ((D_original_motion - D_min_motion) / (D_max_motion - D_min_motion))
    D_norm_motion = np.array(D_norm_motion, dtype=np.uint8)

    JJD_RGB_pose = cv2.applyColorMap(D_norm_pose, cv2.COLORMAP_JET)
    JJD_RGB_motion = cv2.applyColorMap(D_norm_motion, cv2.COLORMAP_JET)

    O_original_pose = np.array(O_original_pose)
    O_original_motion = np.array(O_original_motion)

    # Normalize between [0; 255] and consider (x, y, z) as (R, G, B)
    c_max_pose = np.amax(O_original_pose)
    c_max_motion = np.amax(O_original_motion)

    c_min_pose = np.amin(O_original_pose)
    c_min_motion = np.amin(O_original_motion)
    
    JJO_RGB_pose = np.array(np.floor(255 * ((O_original_pose - c_min_pose) / (c_max_pose - c_min_pose))), dtype=np.uint8)
    JJO_RGB_motion = np.array(np.floor(255 * ((O_original_motion - c_min_motion) / (c_max_motion - c_min_motion))), dtype=np.uint8)

    PF = np.zeros((len(coords_frames), 200,  3), dtype=np.uint8)
    MF = np.zeros((len(coords_frames)-1, 200, 3), dtype=np.uint8)

    for nb_frame in range(len(coords_frames)):
        temp_pf = np.concatenate([JJD_RGB_pose[nb_frame], JJO_RGB_pose[nb_frame]])
        PF[nb_frame] = temp_pf
        if nb_frame < len(coords_frames)-1:
            temp_mf = np.concatenate([JJD_RGB_motion[nb_frame], JJO_RGB_motion[nb_frame]])
            MF[nb_frame] = temp_mf

    PF = np.array(PF)
    MF = np.array(MF)
    SPMF = np.zeros((200, 2*len(coords_frames)-1, 3), dtype=np.uint8)

    frame_index = 0
    for pf, mf in zip(PF, MF):
        SPMF[:, frame_index] = pf
        frame_index += 1
        SPMF[:, frame_index] = mf
        frame_index += 1
    
    # Add the last element of PF to SPMF because the zip function keeps only the elements in common between the two tables. 
    # The PF table has one element more than MF.
    SPMF[:, frame_index] = PF[-1]

    return SPMF

def bvh2GeometricFeaturesV3(frames, joints, ignored_joints, ignored_joints_index):
    """
        Convert BVH informations to RGB image after computing geometric features

        :param bvhParser: parser used to parse the given bvh
        :type bvhParser: BVHParser Class
        :param joints: List of body joints
        :type joints: array of Joint Object
        :param ignored_joints: List of joints we want to ignore
        :type ignored_joints: array of string

        :return: RGB image
        :rtype: numpy array
    """

    # image's width = number of frames, image's height = number of joints - number of ignored joints
    width, height = len(frames), (len(joints)-len(ignored_joints))

    coords_frames = np.zeros((width, height, 3), dtype=np.float32)

    for frame_index, frame in enumerate(frames):
        stack_joints = []
        stack_M = []
        joint_placed = 0
        pixel_placed = 0
        for joint_index in range(0, len(joints)):
            if joint_index in ignored_joints_index:
                stack_joints.append(np.array([None, None, None, None, None]))
                stack_M.append(np.eye(4))
                joint_placed += 6
                continue
            
            if joint_index == 0: # ROOT
                bvh_content = [frame[joint_placed + i] for i in range(6)]
                
                stack_joints.append(bvh_content)
                
                stack_M.append(np.eye(4))
                
                coords_frames[frame_index][pixel_placed] = bvh_content[:3]
            
            else:
                bvh_content = [frame[joint_placed+i] for i in range(6)]

                M_parent = stack_M[joints[joint_index].parent.index]
                Tx, Ty, Tz, Ry, Rx, Rz = stack_joints[joints[joint_index].parent.index]
                
                T = getTranslationMat(Tx, Ty, Tz)

                rot_mat_Y, rot_mat_X, rot_mat_Z = getRotationMatY(Ry), getRotationMatX(Rx), getRotationMatZ(Rz)
                
                R = rot_mat_Y.dot(rot_mat_X).dot(rot_mat_Z)

                M = T.dot(R)
                M = M_parent.dot(M)
                
                local_coordinates = np.concatenate([bvh_content[:3], np.array([1])])
                global_coordinates = M.dot(local_coordinates)

                coords_frames[frame_index][pixel_placed] = global_coordinates[:3]
                
                stack_M.append(M)
                
                stack_joints.append(bvh_content)
            
            joint_placed += 6
            pixel_placed += 1

    # For each frame, compute all JJD (Joint-Joint Distance) and JJO (Joint-Joint Orientation) for Pose Features (PF) and Motion Features (MF)
    D_original_pose = []
    O_original_pose = []
    D_original_motion = []
    O_original_motion = []

    for t in range(len(coords_frames)):
        frame_t = coords_frames[t]
        if t != len(coords_frames)-1:
            next_frame = coords_frames[t+1]

        JJD_pose = []
        JJD_motion = []
        JJO_pose = []
        JJO_motion = []
        root_t = frame_t[0]

        for joint in range(1, len(frame_t)):
            euclidean_distance_pose = math.sqrt( 
                (frame_t[joint][0] - root_t[0])**2 + 
                (frame_t[joint][1] - root_t[1])**2 + 
                (frame_t[joint][2] - root_t[2])**2
            )
            JJD_pose.append(euclidean_distance_pose)
            
            if t != len(coords_frames)-1:
                euclidean_distance_motion = math.sqrt( 
                    (next_frame[joint][0] - root_t[0])**2 + 
                    (next_frame[joint][1] - root_t[1])**2 + 
                    (next_frame[joint][2] - root_t[2])**2 
                )
                JJD_motion.append(euclidean_distance_motion)
            
            orientation_pose = np.array(
                [
                    -frame_t[joint][0] + root_t[0],
                    -frame_t[joint][1] + root_t[1],
                    -frame_t[joint][2] + root_t[2],
                ]
            )
            JJO_pose.append(orientation_pose)

            if t != len(coords_frames)-1:
                orientation_motion = np.array(
                    [
                        -next_frame[joint][0] + root_t[0],
                        -next_frame[joint][1] + root_t[1],
                        -next_frame[joint][2] + root_t[2],
                    ]
                )
                JJO_motion.append(orientation_motion)

        D_original_pose.append(JJD_pose)
        O_original_pose.append(JJO_pose)

        if t != len(coords_frames)-1: 
            D_original_motion.append(JJD_motion)
            O_original_motion.append(JJO_motion)
    
    D_original_pose = np.array(D_original_pose)
    D_original_motion = np.array(D_original_motion)
    
    # Normalize D_original_pose and D_original_motion between [0; 255] and apply JET color map on them to obtain two images with 3 channels (R, G, B)
    D_min_pose = 0
    D_min_motion = 0
    
    D_max_pose = np.amax(D_original_pose)
    D_max_motion = np.amax(D_original_motion)

    D_norm_pose = 255 * ((D_original_pose - D_min_pose) / (D_max_pose - D_min_pose))
    D_norm_pose = np.array(D_norm_pose, dtype=np.uint8)
    
    D_norm_motion = 255 * ((D_original_motion - D_min_motion) / (D_max_motion - D_min_motion))
    D_norm_motion = np.array(D_norm_motion, dtype=np.uint8)

    JJD_RGB_pose = cv2.applyColorMap(D_norm_pose, cv2.COLORMAP_JET)
    # print(JJD_RGB_pose.shape)
    JJD_RGB_motion = cv2.applyColorMap(D_norm_motion, cv2.COLORMAP_JET)
    # print(JJD_RGB_motion.shape)

    O_original_pose = np.array(O_original_pose)
    O_original_motion = np.array(O_original_motion)

    # Normalize between [0; 255] and consider (x, y, z) as (R, G, B)
    c_max_pose = np.amax(O_original_pose)
    c_max_motion = np.amax(O_original_motion)

    c_min_pose = np.amin(O_original_pose)
    c_min_motion = np.amin(O_original_motion)
    
    JJO_RGB_pose = np.array(np.floor(255 * ((O_original_pose - c_min_pose) / (c_max_pose - c_min_pose))), dtype=np.uint8)
    # print(JJO_RGB_pose.shape)
    JJO_RGB_motion = np.array(np.floor(255 * ((O_original_motion - c_min_motion) / (c_max_motion - c_min_motion))), dtype=np.uint8)
    # print(JJO_RGB_motion.shape)

    PF = np.zeros((len(coords_frames), 40,  3), dtype=np.uint8)
    MF = np.zeros((len(coords_frames)-1, 40, 3), dtype=np.uint8)

    for nb_frame in range(len(coords_frames)):
        temp_pf = np.concatenate([JJD_RGB_pose[nb_frame], JJO_RGB_pose[nb_frame]])
        # temp_pf = JJD_RGB_pose[nb_frame]
        PF[nb_frame] = temp_pf
        if nb_frame < len(coords_frames)-1:
            temp_mf = np.concatenate([JJD_RGB_motion[nb_frame], JJO_RGB_motion[nb_frame]])
            # temp_mf = JJD_RGB_motion[nb_frame]
            MF[nb_frame] = temp_mf

    PF = np.array(PF)
    MF = np.array(MF)
    SPMF = np.zeros((40, 2*len(coords_frames)-1, 3), dtype=np.uint8)

    frame_index = 0
    for pf, mf in zip(PF, MF):
        SPMF[:, frame_index] = pf
        frame_index += 1
        SPMF[:, frame_index] = mf
        frame_index += 1
    
    # Add the last element of PF to SPMF because the zip function keeps only the elements in common between the two tables. 
    # The PF table has one element more than MF.
    SPMF[:, frame_index] = PF[-1]
    SPMF = convertRGB2BGR(SPMF)

    return SPMF

def convertRGB2BGR(image):
    """
        Convert given RGB image to BGR image for OpenCV usage.
        Split the 3 channels of input and revert them to obtain BGR image. 

        :param image: RGB image we want to convert
        :type image: numpy array
        
        :return: BGR image
        :rtype: numpy array
    """
    R,G,B = cv2.split(image)
    return cv2.merge([B,G,R])

def xyz2RGBImage(frames, joints, ignored_joints, ignored_joints_index):
    """
        Convert BVH informations to RGB image after computing geometric features

        :param frames: frames contained in the bvh parsed file
        :type frames: numpy array
        :param joints: List of body joints
        :type joints: array of Joint Object
        :param ignored_joints: List of joints we want to ignore
        :type ignored_joints: array of string

        :return: RGB image
        :rtype: numpy array
    """

    coords_frames = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)

    min_coords, max_coords = np.amin(coords_frames), np.amax(coords_frames)

    coords_frames = np.array(255 * ((coords_frames - min_coords) / (max_coords - min_coords)), dtype=np.uint8)
    coords_frames = np.moveaxis(coords_frames, 0, 1)
    
    return coords_frames

def bvh2RGBImage(bvhParser, joints, ignored_joints, config):
    """
        Convert BVH informations to RGB image with Laraba or Ludl techniques

        :param bvhParser: parser used to parse the given bvh
        :type bvhParser: BVHParser Class
        :param joints: List of body joints
        :type joints: array of Joint Class
        :param ignored_joints: List of joints we want to ignore
        :type ignored_joints: array of string
        :param config: parameters of given config file
        :type config: dict

        :return: RGB image
        :rtype: numpy array

    """
    ignored_joints_index = ignoreJoints(bvhParser, config["coordinate"], ignored_joints)
    
    if config["coordinate"] == BOTH:
        height_multiplier = 2
    else:
        height_multiplier = 1

    width, height = bvhParser.nb_frames, len(joints) * height_multiplier

    img_struct = (height, width, 3)

    img_content = np.zeros(img_struct, dtype=np.uint8)


    if config["coordinate"] == TRANSLATION:
        translation_min_max_X, translation_min_max_Y, translation_min_max_Z = get_min_max_translation(bvhParser.frames)
    elif config["coordinate"] == ROTATION:
        rotation_min_max_X, rotation_min_max_Y, rotation_min_max_Z = get_min_max_rotation(bvhParser.frames)
    else:
        translation_min_max_X, translation_min_max_Y, translation_min_max_Z = get_min_max_translation(bvhParser.frames)
        rotation_min_max_X, rotation_min_max_Y, rotation_min_max_Z = get_min_max_rotation(bvhParser.frames)
    
    for nb_frame, frame in enumerate(bvhParser.frames):
        if config["coordinate"] == ROTATION:
            joint_placed = 3
        else:
            joint_placed = 0
        for nb_joint in range(0, height):
            if nb_joint in ignored_joints_index:
                joint_placed += 3
                continue
            try:
                if config["encoding"] == "LUDL":
                    img_content[nb_joint][nb_frame] = calculate_ludl_value(frame, joint_placed)

                if config["encoding"] == "LARABA":
                    if config["coordinate"] == ROTATION:
                        img_content[nb_joint][nb_frame] = calculate_laraba_value_rotation(frame, joint_placed, rotation_min_max_X, rotation_min_max_Y, rotation_min_max_Z)
                    if config["coordinate"] == TRANSLATION:
                        img_content[nb_joint][nb_frame] = calculate_laraba_value_translation(frame, joint_placed, translation_min_max_X, translation_min_max_Y, translation_min_max_Z)
                    if config["coordinate"] == HYBRID:
                        if nb_joint == 0:
                            img_content[nb_joint][nb_frame] = calculate_laraba_value_translation(frame, joint_placed, translation_min_max_X, translation_min_max_Y, translation_min_max_Z)
                        else:
                            img_content[nb_joint][nb_frame]= calculate_laraba_value_rotation(frame, joint_placed, rotation_min_max_X, rotation_min_max_Y, rotation_min_max_Z)
                    if config["coordinate"] == BOTH:
                        if nb_joint % 2 == 0:
                            img_content[nb_joint][nb_frame] = calculate_laraba_value_translation(frame, joint_placed, translation_min_max_X, translation_min_max_Y, translation_min_max_Z)
                        else:
                            img_content[nb_joint][nb_frame] = calculate_laraba_value_rotation(frame, joint_placed, rotation_min_max_X, rotation_min_max_Y, rotation_min_max_Z)
            except:
                pass

            if (config["coordinate"] == BOTH) or (config["coordinate"] == HYBRID and nb_joint == 0):
                joint_placed += 3
            else:
                joint_placed += 6

    #Delete black rows
    img_content = np.delete(img_content, ignored_joints_index, axis=0)

    img_content = convertRGB2BGR(img_content)

    return img_content

def saveImage(img,path, filename, encoding_method, coordinate, informations, enhancement=False):
    """ 
        Save the given image in a file  

        :param img: image we want to save
        :type img: numpy array
        :param path: path to write the new file
        :type path: string
        :param filename: name of the BVH file
        :type filename: string
        :param encoding_method: method used to encode BVH informations in pixels (appears in image name)
        :type encoding_method: string
        :param coordinate: coordinates used for the resulting image
        :type coordinate: int
        :param informations: Additional informations given at launch to appears in images name
        :type informations: string
        :param enhancement: if true, apply CLAHE (Constrast Limiting Adaptive Histogram Equalization) to adjust contrast and colors
        :type enhancement: bool
    """
    
    # Change coordinate used to string to put it in the image name
    if coordinate == TRANSLATION:
        coord = "TRANSLATION"
    elif coordinate == ROTATION:
        coord = "ROTATION"
    elif coordinate == BOTH:
        coord = "BOTH"
    else:
        coord = "HYBRID"
    
    if enhancement:
        # Enhancement cannot be applied on multiples channels.
        # To apply CLAHE on a RGB image, we need to convert it to HSV (Hue, Saturation and Value). 
        # In this mode, all the values are contained in one channel.
        # So, we can apply the CLAHE algorithm
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H,S,V = cv2.split(img)
        ahe = cv2.createCLAHE(clipLimit=CLIPLIMIT, tileGridSize=(8,8))
        V = ahe.apply(V)
        img = cv2.merge([H,S,V])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        print("[+] - Saving image in '" + path + "/" + filename[:-4] + "_" + encoding_method.upper() + "_" + coord + "_" + informations.upper() + "_ENHANCED.png'")
        cv2.imwrite("./img/"+path.split("/")[-1] + "/" + filename[:-4] + "_" + encoding_method.upper() + "_" + coord + "_" + informations.upper() + "_ENHANCED.png", img)
    else:
        print("[+] - Saving image in '" + path + "/" + filename[:-4] + "_" + encoding_method.upper() + "_" + coord + "_" + informations.upper() + ".png'")
        cv2.imwrite("./img/"+path.split("/")[-1] + "/" + filename[:-4] + "_" + encoding_method.upper() + "_" + coord + "_" + informations.upper() + ".png", img)

def saveImages(images, path, filename, informations):
    """ 
        Save the given images in multiple files 

        :param images: images we want to save
        :type img: list of numpy arrays
        :param path: path to write the new file
        :type path: string
        :param filename: name of the BVH file
        :type filename: string
        :param informations: Additional informations given at launch to appears in images name
        :type informations: string
    """
    for img_index in range(len(images)):
        print("[+] - Saving image in '" + path + "/" + filename[:-4] + "/" + filename[:-4] + "_" + informations.upper() + "_" + str(img_index) + ".png'")
        cv2.imwrite("./img/"+path.split("/")[-1] + "/" + filename[:-4] + "/" + filename[:-4] + "_" + informations.upper() + "_" + str(img_index) + ".png", images[img_index])

def applyInterpolation(base_img, dim, path, filename, encoding_method, interpolation_method, coordinate, informations, enhancement=False):
    """ 
        Apply an interpolation method on the original image to resize it. 
        And save the resulting image in a file  

        :param base_img: image we want to save
        :type img: numpy array
        :param dim: dimensions of the new image
        :type dim: tuple(int, int)
        :param path: path to write the new file
        :type path: string
        :param filename: name of the BVH file
        :type filename: string
        :param encoding_method: method used to encode BVH informations in pixels (appears in image name)
        :type encoding_method: string
        :param interpolation_method: Interpolation method used to resize the image
        :type interpolation: string
        :param coordinate: coordinates used for the resulting image
        :type coordinate: int
        :param informations: Additional informations given at launch to appears in images name
        :type coordinate: string
        :param enhancement: if true, apply CLAHE (Constrast Limiting Adaptive Histogram Equalization) to adjust contrast and colors
        :type enhancement: bool
    """

    if interpolation_method.upper() == "NEAREST":
        interpolation=cv2.INTER_NEAREST
    elif interpolation_method.upper() == "BICUBIC":
        interpolation=cv2.INTER_CUBIC
    else:
        print(WARNING + "[!] - Given interpolation technique is not correct, must be 'NEAREST' or 'BICUBIC'" + RESET)
        return
    # dim=(32,32)
    new_img = cv2.resize(base_img, dim, interpolation=interpolation)
    saveImage(new_img, path, filename, encoding_method.upper()+"_"+interpolation_method.upper(), coordinate, informations, enhancement)

def ignoreJoints(bvhParser, coordinate=BOTH, ignored_joints=[]):
    """ 
        Find joint indexes of ignored joints.
        If both coordinates are used (translation and rotation), indexes are doubled

        :param bvhParser: bvhParser object
        :type bvhParser: BVHParser Class
        :param coordinate: coordinates used to convert bvh to image
        :type path: string
        :param ignored_joints: names of ignored joints
        :type filename: array
        
        :return: indexes corresponding to the list of ignored joints 
        :rtype: array
    """

    if coordinate == BOTH:
        ignored_joints_index_pos = [2*bvhParser.joints[j].index for j in ignored_joints]
        ignored_joints_index_rot = [2*bvhParser.joints[j].index+1 for j in ignored_joints]
        ignored_joints_index = [*ignored_joints_index_pos, *ignored_joints_index_rot]
    else:
        ignored_joints_index = [bvhParser.joints[j].index for j in ignored_joints]
    
    return ignored_joints_index

def parseFile(path, filename):
    """
        Create a parser and use it to parse the given file

        :param path: Path of the BVH file we want to parse
        :type path: string
        :param filename: Name of the BVH file
        :type filename: string

        :return: bvhParser object
        :rtype: BVHParser Class
    """

    bvhParser = bvh_parser.BVHParser()
    bvhParser.parse(path+"/"+filename)
    return bvhParser

def checkFormat(frames, joints):
    """
        Check if BVH file is correctly formed
        One BVH frame must have 6 coordinates (Tx, Ty, Tz, Rx, Ry, Rz) per joints
    """

    if len(frames[0]) != 6*len(joints):
        print(WARNING+"[!] - Format not correct"+RESET)

def main(path, filename, config):
    """
        Main function

        1) Parse the file
        2) Get the joints list
        3) Check the BVH file
        4) Encode BVH informations with the technique specified in the config file
        5) Save the image in the file
    """

    multiple_images = False

    bvhParser = parseFile(path, filename)

    joints = bvhParser.get_joints_list()
    
    checkFormat(bvhParser.frames, joints)

    if config["encoding"] == "GEOMETRIC":
        ignored_joints_index = ignoreJoints(bvhParser, 0, IGNORED_JOINTS)

        # --- Use version 1 of geometric features (Pham) ---
        image = bvh2geometric.bvh2GeometricFeatures(bvhParser.frames, joints, IGNORED_JOINTS, ignored_joints_index)

        # --- Use version 2 of geometric features (Root only) ---
        # image = bvh2GeometricFeaturesCustom(bvhParser.frames, joints, IGNORED_JOINTS, ignored_joints_index, joints_we_want)
        
        # --- Use version 3 of geometric features (Multiple joint for computation => joints_we_want array) ---
        
        # Define here the joint index you want to use
        # joints_we_want = np.array([4, 7, 12, 16, 20], dtype=np.uint8) # head, feet, hands
        joints_we_want = np.array([0, 13, 16, 17, 20], dtype=np.uint8) # root, shoulders, hands
        # joints_we_want = np.array([0, 14, 16, 18, 20], dtype=np.uint8) # root, arm, hands
        
        # image = bvh2geometric.bvh2GeometricFeaturesV2(bvhParser.frames, joints, IGNORED_JOINTS, ignored_joints_index)
    elif config["encoding"] == "MULTIPLE":
        # Set this flag to True to call a separate saving method
        multiple_images = True

        ignored_joints_index = ignoreJoints(bvhParser, 0, IGNORED_JOINTS)
        
        images = bvh2geometric.bvh2MultipleImages(bvhParser.frames, joints, IGNORED_JOINTS, ignored_joints_index)
    elif config["encoding"] == "LARABA" or config["encoding"] == "LUDL":
        image = bvh2RGBImage(bvhParser, joints, IGNORED_JOINTS, config)
    elif config["encoding"]== "XYZ":
        ignored_joints_index = ignoreJoints(bvhParser, 0, IGNORED_JOINTS)
        
        image = xyz2RGBImage(bvhParser.frames, joints, IGNORED_JOINTS, ignored_joints_index)
    
    # Check if directory exists, if not create it
    path_to_img = Path("./img/"+path.split("/")[-1])
    path_to_img.mkdir(exist_ok=True)

    if multiple_images:
        # In this case, create also a directory corresponding to the filename inside the class folder.
        # This directory will contain several images
        path_to_img_dir = Path("./img/"+path.split("/")[-1]+"/"+filename[:-4])
        path_to_img_dir.mkdir(exist_ok=True)

        saveImages(images, path, filename, config["details"])

    else:
        if config["interpolation"] == True:
            applyInterpolation(image, config["resize_dim"], path, filename, config["encoding"], config["interpolation_method"], config["coordinate"], config["details"], config["enhancement"])
        else:
            saveImage(image, path, filename, config["encoding"], config["coordinate"], config["details"], config["enhancement"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Command line arguments
    available_arguments = [
        {
            "name": "--test",
            "help": "Test encoding techniques on sample images rather than on the entire dataset",
            "action": "store_true",
            "default": False,
            "required": False
        },
        {
            "name": "--config",
            "help": "Specify the config file to use to convert bvh files in images",
            "action": "store",
            "required": True,
            "default": "./config/geometric.cfg"
        },
        {
            "name": "--details",
            "help": "Add additional text to image filenames",
            "action": "store",
            "default": "",
            "required": False
        }        
    ]

    for argument in available_arguments:
        parser.add_argument(
            argument["name"],
            help=argument["help"],
            action=argument["action"],
            default=argument["default"],
            required=argument["required"]
        )
    
    # parse command line arguments
    args = parser.parse_args()

    # parse config file specified in command line
    config = parseConfigFile(args)

    if args.test:
        DATADIR = "./data/Others"

    # For all directories in the DATADIR directory, convert BVH files to images
    for dirpath, dirnames, files in os.walk(DATADIR):
        print("[+] - Exploring directory: " + dirpath)

        samples = files

        for file_name in samples:
            main(dirpath, file_name, config)