import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse
import math
import os
import random
import time
import threading
from pathlib import Path

import cv2
import numpy as np

import lib.bvh2geometric as bvh2geometric
from src.bvh import bvh_parser
from src.constants import *
from src.utils import *

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
        print("[+] - Saving image in '" + path + "/" + filename[:-4] + "_" + informations + "/" + filename[:-4] + "_" + informations + "_" + str(img_index) + ".png'")
        cv2.imwrite("../img/"+path.split("/")[-1] + "/" + filename[:-4] + "_" + informations + "/" + filename[:-4] + "_" + informations + "_" + str(img_index) + ".png", images[img_index])

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

def oversampling(path, filename):
    """
        Main function

        1) Parse the file
        2) Get the joints list
        3) Check the BVH file
        4) Encode BVH informations with the technique specified in the config file
        5) Save the image in the file
    """
    print(filename)
    timestamp = str(time.time())
    multiple_images = False

    bvhParser = parseFile(path, filename)

    joints = bvhParser.get_joints_list()
    
    checkFormat(bvhParser.frames, joints)

    # Set this flag to True to call a separate saving method
    multiple_images = True

    ignored_joints_index = ignoreJoints(bvhParser, 0, IGNORED_JOINTS)
    
    images = bvh2geometric.oversample_bvh2MultipleImages(bvhParser.frames, joints, IGNORED_JOINTS, ignored_joints_index)
    
    # Check if directory exists, if not create it
    path_to_img = Path("../img/"+path.split("/")[-1])
    path_to_img.mkdir(exist_ok=True)

    # In this case, create also a directory corresponding to the filename inside the class folder.
    # This directory will contain several images
    path_to_img_dir = Path("../img/"+path.split("/")[-1]+"/"+filename[:-4]+"_"+timestamp)
    path_to_img_dir.mkdir(exist_ok=True)

    saveImages(images, path, filename, timestamp)

if __name__ == "__main__":

    DATADIR = "./data_to_oversample/"
    desired_size = 550

    # For all directories in the DATADIR directory, convert BVH files to images
    for dirpath, dirnames, files in os.walk(DATADIR):
        print("[+] - Exploring directory: " + dirpath)
        actual_size = len(files)
        if actual_size < 38: #min size of InHARD dataset
            continue
        if actual_size >= desired_size: continue
        images_to_crop = desired_size - actual_size
        for i in range(images_to_crop):
            filename = random.choice(files)
            oversampling(dirpath, filename)
