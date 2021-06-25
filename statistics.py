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
import json
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

statistics = {}

def printProgressBar(done, total):
    bar_len = 60
    filled_len = int(round(bar_len * done / float(total)))

    percentages = round(100.0 * done / float(total), 1)
    bar = str(done) + "/" + str(total) + " " + "█" * filled_len + "." * (bar_len - filled_len)

    print(bar, end="\r")

def calc_energy(coords, rho, step):
    energies = [0 for i in range(len(coords[0]))]
    F = len(coords)-1
    for f in range(0, F-step, step):
        current_frame = coords[f]
        next_frame = coords[f+1]
        for n in range(len(coords[0])):
            energy = np.linalg.norm(next_frame[n] - current_frame[n])
            energies[n] += energy

    energies_array = np.array(energies)
    return np.sum(energies_array)

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

def main(path, filename):
    """
        Main function
    """

    class_name = path.split("/")[-1]

    bvhParser = parseFile(path, filename)

    joints = bvhParser.get_joints_list()
    
    checkFormat(bvhParser.frames, joints)

    ignored_joints_index = ignoreJoints(bvhParser, 0, IGNORED_JOINTS)
    
    coords_frames = getWorldCoordinates(bvhParser.frames, joints, IGNORED_JOINTS, ignored_joints_index)

    total_energy = calc_energy(coords_frames, 1, 1)

    if not class_name in statistics:
        statistics[class_name] = {}
    filename = filename.split(".")[0]
    statistics[class_name][filename] = {"number_of_frames": len(coords_frames), "total_energy": total_energy}

if __name__ == "__main__":
    DATADIR = "./data/"
    conversions_done = 0
    total = 0
    class_we_want = ["Consult_sheets", "Picking_in_front", "Picking_left", "Take_screwdriver"]

    # For all directories in the DATADIR directory, convert BVH files to images
    for dirpath, dirnames, files in os.walk(DATADIR):
        print("[+] - Exploring directory: " + dirpath + "\n")
        if len(files) == 0: continue
        total += len(files)
        printProgressBar(conversions_done, total)
        class_name = dirpath.split("/")[-1]
        if not class_name in class_we_want:
            conversions_done += len(files)
            continue

        for file_name in files:
            main(dirpath, file_name)
            conversions_done += 1
            printProgressBar(conversions_done, total)

    json_string = json.dumps(statistics, indent=4)
    print(json_string)

    with open("./statistics.json", "w") as json_file:
        json_file.write(json_string)
        