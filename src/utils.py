import math, configparser, numpy as np
from .constants import *

def parseArguments(args):
    """ 
        Parse command line arguments given at launch

        :param args: The given arguments
        :type args: argparse Object 
        :return: Main parameters of the program
        :rtype: dict
    """
    entries = {}

    entries["original"] = args.original
    entries["nearest"] = args.nearest
    entries["bicubic"] = args.bicubic
    
    entries["enhancement"] = args.enhancement
    entries["additional"] = args.additional
    
    entries["ludl"] = False
    entries["laraba"] = False

    for name in args.transformation:
        if name.lower() == "ludl":
            entries["ludl"] = True
        if name.lower() == "laraba":
            entries["laraba"] = True
    
    entries["coordinate"] = ROTATION
    
    if "TR" in args.coordinate or "RT" in args.coordinate:
        entries["coordinate"] = BOTH
    elif "T" in args.coordinate:
        entries["coordinate"] = TRANSLATION
    elif "R" in args.coordinate:
        entries["coordinate"] = ROTATION
    elif "H" in args.coordinate:
        entries["coordinate"] = HYBRID
    else:
        print(WARNING + "[!] - Problem with coordinate argument" + RESET)
        exit()

    return entries

def parseConfigFile(args):
    """ 
        Parse config file given in argument at launch 

        :param args: The given arguments
        :type args: Argparse Class 

        :return: Main parameters of the program
        :rtype: dict
    """

    config = configparser.ConfigParser()
    config.read(args.config)

    entries = {}

    entries["resize_dim"] = (config["IMAGE"].getint("width"), config["IMAGE"].getint("height"))
    entries["scale_factor"] = config["IMAGE"].getfloat("scale_factor")

    entries["interpolation"] = config["INTERPOLATION"].getboolean("interpolation")
    entries["interpolation_method"] = config["INTERPOLATION"]["method"].upper()
    
    entries["enhancement"] = config["ENHANCEMENT"].getboolean("enhancement")
    entries["clip_limit"] = config["ENHANCEMENT"].getint("clip_limit")

    entries["encoding"] = config["SCRIPT"]["encoding"].upper()
    
    if "TR" in config["SCRIPT"]["coordinate"] or "RT" in config["SCRIPT"]["coordinate"]:
        entries["coordinate"] = BOTH
    elif "T" in config["SCRIPT"]["coordinate"]:
        entries["coordinate"] = TRANSLATION
    elif "R" in config["SCRIPT"]["coordinate"]:
        entries["coordinate"] = ROTATION
    elif "H" in config["SCRIPT"]["coordinate"]:
        entries["coordinate"] = HYBRID
    else:
        print(WARNING + "[!] - Problem with coordinate argument in config" + RESET)
        exit()

    entries["details"] = args.details

    return entries

def get_min_max_rotation(frames):
    """ 
        Get the minimum and maximum of rotation coordinates for the 3 axis (x, y, z) in all the frames 

        :param frames: The frames
        :type frames: numpy array 
        
        :return: Three tuples (one per axis) with the minimum and maximum rotation coordinates
        :rtype: tuple(float, float)
    """

    X_rotation = []
    Y_rotation = []
    Z_rotation = []

    for _, frame in enumerate(frames):
        for joint in range(0, len(frame), 6):
            for pixel in range(0, 3):
                try:
                    value = frame[joint + 3 + pixel]
                except IndexError:
                    continue
                if pixel == 0:
                    Y_rotation.append(value)
                elif pixel == 1:
                    X_rotation.append(value)
                else:
                    Z_rotation.append(value)

    return (min(X_rotation), max(X_rotation)), (min(Y_rotation), max(Y_rotation)), (min(Z_rotation), max(Z_rotation))

def get_min_max_translation(frames):
    """ 
        Get the minimum and maximum of translation coordinates for the 3 axis (x, y, z) in all the frames 

        :param frames: The frames
        :type frames: numpy array 
        
        :return: Three tuples (one per axis) with the minimum and maximum translation coordinates
        :rtype: tuple(float, float)
    """
    
    X_translation = []
    Y_translation = []
    Z_translation = []

    for _, frame in enumerate(frames):
        for joint in range(0, len(frame), 6):
            for pixel in range(0, 3):
                try:
                    value = frame[joint + pixel]
                except IndexError:
                    continue
                if pixel == 0:
                    X_translation.append(value)
                elif pixel == 1:
                    Y_translation.append(value)
                else:
                    Z_translation.append(value)

    return (min(X_translation), max(X_translation)), (min(Y_translation), max(Y_translation)), (min(Z_translation), max(Z_translation))

def calculate_ludl_value(frame, joint):
    return [
        255 * (frame[joint + 1] * (1.0 / 360.0) + 0.5),
        255 * (frame[joint] * (1.0 / 360.0) + 0.5),
        255 * (frame[joint + 2] * (1.0 / 360.0) + 0.5)
    ]

def calculate_laraba_value_rotation(frame, joint, min_max_X, min_max_Y, min_max_Z):
    """ 
        Calculate RGB values with Laraba's method for rotation coordinates 

        :param frame: The current frame
        :type frame: numpy array 
        :param joint: The current joint index
        :type joint: int
        :param min_max_X: Minimum and maximum rotation coordinates around X axis
        :type min_max_X: tuple(float, float)
        :param min_max_Y: Minimum and maximum rotation coordinates around Y axis
        :type min_max_Y: tuple(float, float)
        :param min_max_Z: Minimum and maximum rotation coordinates around Z axis
        :type min_max_Z: tuple(float, float)
        
        :return: RGB values corresponding respectively to the x, y and z rotation coordinates
        :rtype: array[float, float, float]
    """

    # Read first the coordinate at indice 1 because rotations are stored in the order: YXZ 
    return [
        255 * ((frame[joint + 1] - min_max_X[0]) / (min_max_X[1] - min_max_X[0])),
        255 * ((frame[joint] - min_max_Y[0]) / (min_max_Y[1] - min_max_Y[0])),
        255 * ((frame[joint + 2]- min_max_Z[0]) / (min_max_Z[1] - min_max_Z[0]))
    ]

def calculate_laraba_value_translation(frame, joint, min_max_X, min_max_Y, min_max_Z):
    """ 
        Calculate RGB values with Laraba's method for translation coordinates 

        :param frame: The current frame
        :type frame: numpy array 
        :param joint: The current joint index
        :type joint: int
        :param min_max_X: Minimum and maximum translation coordinates along X axis
        :type min_max_X: tuple(float, float)
        :param min_max_Y: Minimum and maximum translation coordinates along Y axis
        :type min_max_Y: tuple(float, float)
        :param min_max_Z: Minimum and maximum translation coordinates along Z axis
        :type min_max_Z: tuple(float, float)
        
        :return: RGB values corresponding respectively to the x, y and z translation coordinates
        :rtype: array[float, float, float]
    """

    return [
        255 * (((frame[joint] * SCALE_FACTOR) - min_max_X[0]) / (min_max_X[1] - min_max_X[0])),
        255 * (((frame[joint + 1] * SCALE_FACTOR) - min_max_Y[0]) / (min_max_Y[1] - min_max_Y[0])),
        255 * (((frame[joint + 2] * SCALE_FACTOR) - min_max_Z[0]) / (min_max_Z[1] - min_max_Z[0]))
    ]

def getRotationMatX(angle):
    """ 
        Calculate the rotation matrix corresponding to a rotation around X axis for the given angle  

        :param angle: Rotation angle
        :type angle: float
        
        :return: Rotation matrix corresponding to a rotation of "angle" degree around X axis
        :rtype: numpy array
    """

    angle = np.deg2rad(angle)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, math.cos(angle), -math.sin(angle), 0],
            [0, math.sin(angle), math.cos(angle), 0],
            [0, 0, 0, 1]
        ]
    )

def getRotationMatY(angle):
    """ 
        Calculate the rotation matrix corresponding to a rotation around Y axis for the given angle  

        :param angle: Rotation angle
        :type angle: float
        
        :return: Rotation matrix corresponding to a rotation of "angle" degree around Y axis
        :rtype: numpy array
    """
    
    angle = np.deg2rad(angle)
    return np.array(
        [
            [math.cos(angle), 0, math.sin(angle), 0],
            [0, 1, 0, 0],
            [-math.sin(angle), 0, math.cos(angle), 0],
            [0, 0, 0, 1]
        ]
    )

def getRotationMatZ(angle):
    """ 
        Calculate the rotation matrix corresponding to a rotation around Z axis for the given angle  

        :param angle: Rotation angle
        :type angle: float
        
        :return: Rotation matrix corresponding to a rotation of "angle" degree around Z axis
        :rtype: numpy array
    """

    angle = np.deg2rad(angle)
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0, 0],
            [math.sin(angle), math.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    )

def getTranslationMat(Tx, Ty, Tz):
    """ 
        Calculate the translation matrix for the given translations along the 3 axis (x, y, z)  

        :param Tx: Translation along X axis
        :type Tx: float
        :param Ty: Translation along Y axis
        :type Ty: float
        :param Tz: Translation along Z axis
        :type Tz: float
      
        :return: Rotation matrix corresponding to a rotation of "angle" degree around X axis
        :rtype: numpy array
    """
    
    return np.array(
        [
            [1, 0, 0, Tx],
            [0, 1, 0, Ty],
            [0, 0, 1, Tz],
            [0, 0, 0, 1]
        ]
    )