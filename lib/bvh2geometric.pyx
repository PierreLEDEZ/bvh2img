# distutils: language=c++
# cython: cdivision=True
# cython: language_level=3

import numpy as np, cv2, time
cimport numpy as np
cimport cython
from cython.parallel import prange

UINT=np.uint8
INT = np.int64
FLOAT=np.float32

ctypedef np.uint8_t UINT_t
ctypedef np.int64_t INT_t
ctypedef np.float32_t FLOAT_t

cdef np.ndarray space_2D = np.array(
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
    dtype=UINT
)

cdef np.ndarray space_3D = np.array(
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
    dtype=UINT
)

cdef np.ndarray space_2D_v2 = np.array(
    [
        [0, 1],
        [0, 2],
        [1, 2]
    ],
    dtype=UINT
)

cdef np.ndarray space_3D_v2 = np.array(
    [
        [2, 3, 4],
        [1, 3, 4],
        [0, 3, 4]
    ],
    dtype=UINT
)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[FLOAT_t, ndim=2] getRotationMatX(float angle):
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
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ],
        dtype=FLOAT
    )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[FLOAT_t, ndim=2] getRotationMatY(float angle):
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
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ],
        dtype=FLOAT
    )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[FLOAT_t, ndim=2] getRotationMatZ(float angle):
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
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        dtype=FLOAT
    )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[FLOAT_t, ndim=2] getTranslationMat(float Tx, float Ty, float Tz):
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
        ],
        dtype=FLOAT
    )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[UINT_t, ndim=3] convertRGB2BGR(np.ndarray[UINT_t, ndim=3] image):
    """
        Convert given RGB image to BGR image for OpenCV usage.
        Split the 3 channels of input and revert them to obtain BGR image. 

        :param image: RGB image we want to convert
        :type image: numpy array
        
        :return: BGR image
        :rtype: numpy array
    """
    cdef np.ndarray R, G, B
    R,G,B = cv2.split(image)
    return cv2.merge([B,G,R])

from libcpp.vector cimport vector
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[FLOAT_t, ndim=3] getWorldCoordinates(np.ndarray[FLOAT_t, ndim=2] frames, list joints, list ignored_joints, list ignored_joints_index):
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
    cdef int width = len(frames)
    cdef int height = len(joints)-len(ignored_joints)

    cdef np.ndarray[FLOAT_t, ndim=3] coords_frames = np.empty((width, height, 3), dtype=FLOAT)
    cdef float [:,:,:] coords_frames_view = coords_frames


    cdef:
        list stack_joints, stack_M, bvh_content
        int joint_placed, pixel_placed, frame_index, joint_index, i
        np.ndarray[FLOAT_t, ndim=1] frame
        np.ndarray[FLOAT_t, ndim=2] rot_mat_X, rot_mat_Y, rot_mat_Z, T, R
        np.ndarray M
        np.ndarray local_coordinates, global_coordinates
        float Tx, Ty, Tz, Ry, Rx, Rz
        float initial_Yrotation = 0.0

    for frame_index, frame in enumerate(frames):
        stack_joints = []
        stack_M = []
        joint_placed = 0
        pixel_placed = 0
        for joint_index in xrange(0, len(joints)):
            if joint_index in ignored_joints_index:
                stack_joints.append(np.array([None, None, None, None, None]))
                stack_M.append(np.eye(4))
                joint_placed += 6
                continue
            
            if joint_index == 0: # ROOT
                bvh_content = [frame[joint_placed + i] for i in xrange(6)]
                
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
                
                stack_joints.append(bvh_content)
                
                stack_M.append(np.eye(4))
                for i in xrange(3):
                    coords_frames_view[frame_index][pixel_placed][i] = bvh_content[i]
            
            else:
                bvh_content = [frame[joint_placed+i] for i in xrange(6)]

                M_parent = stack_M[joints[joint_index].parent.index]
                Tx, Ty, Tz, Ry, Rx, Rz = stack_joints[joints[joint_index].parent.index]
                
                T = getTranslationMat(Tx, Ty, Tz)

                rot_mat_Y, rot_mat_X, rot_mat_Z = getRotationMatY(Ry), getRotationMatX(Rx), getRotationMatZ(Rz)
                
                R = rot_mat_Y.dot(rot_mat_X).dot(rot_mat_Z)

                M = T.dot(R)
                M = M_parent.dot(M)
                
                local_coordinates = np.concatenate([bvh_content[:3], np.array([1])])
                global_coordinates = M.dot(local_coordinates)
                for i in xrange(3):
                    coords_frames_view[frame_index][pixel_placed][i] = global_coordinates[i]
                
                stack_M.append(M)
                
                stack_joints.append(bvh_content)
            
            joint_placed += 6
            pixel_placed += 1

    return coords_frames

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[UINT_t, ndim=3] bvh2GeometricFeatures(np.ndarray[FLOAT_t, ndim=2] frames, list joints, list ignored_joints, list ignored_joints_index):
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

    cdef np.ndarray[FLOAT_t, ndim=3] coords_frames = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)
    cdef float [:,:,:] coords_frames_view = coords_frames
    # For each frame, compute all JJD (Joint-Joint Distance) and JJO (Joint-Joint Orientation) for Pose Features (PF) and Motion Features (MF)
    D_original_pose = []
    O_original_pose = []
    D_original_motion = []
    O_original_motion = []

    cdef:
        np.ndarray frame_t, next_frame, orientation_motion, orientation_pose, tmp_j, tmp_k, tmp_j_1, tmp_k_1
        np.ndarray[FLOAT_t, ndim=1] JJD_pose
        np.ndarray[FLOAT_t, ndim=1] JJD_motion
        np.ndarray[FLOAT_t, ndim=2] JJO_pose
        np.ndarray[FLOAT_t, ndim=2] JJO_motion
        float [:] JJD_pose_view
        float [:] JJD_motion_view
        float [:,:] JJO_pose_view
        float [:,:] JJO_motion_view
        int t, joint_J, joint_K, i
        int N=coords_frames.shape[0]
        int M=coords_frames.shape[1]
        float euclidean_distance_motion, euclidean_distance_pose
    
    indexes = [[joint_J, joint_K] for joint_J in xrange(M-1) for joint_K in xrange(joint_J+1, M)]
    for t in xrange(N):
        frame_t = coords_frames[t]
        if t != N-1:
            next_frame = coords_frames[t+1]
        JJD_pose = np.empty((210), dtype=FLOAT)
        JJD_motion = np.empty((210), dtype=FLOAT)
        JJO_pose = np.empty((210, 3), dtype=FLOAT)
        JJO_motion = np.empty((210, 3), dtype=FLOAT)
        JJD_pose_view = JJD_pose
        JJD_motion_view = JJD_motion
        JJO_pose_view = JJO_pose
        JJO_motion_view = JJO_motion
        i = 0
        for joint_J, joint_K in indexes:
        
            tmp_j = frame_t[joint_J]
            tmp_k = frame_t[joint_K]

            euclidean_distance_pose = np.linalg.norm(tmp_j - tmp_k)
            JJD_pose[i] = euclidean_distance_pose
            if t != N-1:
                tmp_k_1 = next_frame[joint_K]
                euclidean_distance_motion = np.linalg.norm(tmp_j - tmp_k_1)
                JJD_motion[i] = euclidean_distance_motion
            
            orientation_pose = tmp_k - tmp_j

            JJO_pose[i] = orientation_pose

            if t != N-1:
                tmp_j_1 = next_frame[joint_J]
                orientation_motion = tmp_k - tmp_j_1
                JJO_motion[i] = orientation_motion
            
            i+=1

        D_original_pose.append(JJD_pose)
        O_original_pose.append(JJO_pose)

        if t != N-1: 
            D_original_motion.append(JJD_motion)
            O_original_motion.append(JJO_motion)

    D_original_pose = np.array(D_original_pose)
    D_original_motion = np.array(D_original_motion)

    # Normalize D_original_pose and D_original_motion between [0; 255] and apply JET color map on them to obtain two images with 3 channels (R, G, B)
    cdef int D_min_pose = 0
    cdef int D_min_motion = 0
    
    D_max_pose = np.amax(D_original_pose)
    D_max_motion = np.amax(D_original_motion)

    cdef np.ndarray D_norm_pose = 255 * ((D_original_pose - D_min_pose) / (D_max_pose - D_min_pose))
    D_norm_pose = np.array(D_norm_pose, dtype=UINT)
    
    cdef np.ndarray D_norm_motion = 255 * ((D_original_motion - D_min_motion) / (D_max_motion - D_min_motion))
    D_norm_motion = np.array(D_norm_motion, dtype=UINT)

    cdef np.ndarray JJD_RGB_pose = cv2.applyColorMap(D_norm_pose, cv2.COLORMAP_JET)
    cdef np.ndarray JJD_RGB_motion = cv2.applyColorMap(D_norm_motion, cv2.COLORMAP_JET)

    O_original_pose = np.array(O_original_pose)
    O_original_motion = np.array(O_original_motion)

    # Normalize between [0; 255] and consider (x, y, z) as (R, G, B)
    c_max_pose = np.amax(O_original_pose)
    c_max_motion = np.amax(O_original_motion)

    c_min_pose = np.amin(O_original_pose)
    c_min_motion = np.amin(O_original_motion)
    
    JJO_RGB_pose = np.array(np.floor(255 * ((O_original_pose - c_min_pose) / (c_max_pose - c_min_pose))), dtype=UINT)
    JJO_RGB_motion = np.array(np.floor(255 * ((O_original_motion - c_min_motion) / (c_max_motion - c_min_motion))), dtype=UINT)

    cdef np.ndarray[UINT_t, ndim=3] PF = np.empty((len(coords_frames), 420,  3), dtype=UINT)

    cdef np.ndarray[UINT_t, ndim=3] MF = np.empty((len(coords_frames)-1, 420, 3), dtype=UINT)

    for nb_frame in xrange(len(coords_frames)):
        temp_pf = np.concatenate([JJD_RGB_pose[nb_frame], JJO_RGB_pose[nb_frame]])
        PF[nb_frame] = temp_pf

        if nb_frame<len(coords_frames)-1:
            temp_mf = np.concatenate([JJD_RGB_motion[nb_frame], JJO_RGB_motion[nb_frame]])
            MF[nb_frame] = temp_mf

    cdef np.ndarray[UINT_t, ndim=3] SPMF = np.empty((420, 2*len(coords_frames)-1, 3), dtype=UINT)
    # cdef np.ndarray[UINT_t, ndim=3] SPMF = np.empty((420, len(coords_frames), 3), dtype=UINT)
    cdef int frame_index = 0
    cdef np.ndarray pf, mf

    for pf, mf in zip(PF, MF):
        SPMF[:, frame_index] = pf
        frame_index += 1
        SPMF[:, frame_index] = mf
        frame_index += 1

    # for pf in PF:
    #     SPMF[:, frame_index] = pf
    #     frame_index += 1

    # Add the last element of PF to SPMF because the zip function keeps only the elements in common between the two tables. 
    # The PF table has one element more than MF.
    SPMF[:, frame_index] = PF[len(PF)-1]
    # SPMF = convertRGB2BGR(SPMF)

    return SPMF

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[UINT_t, ndim=3] bvh2GeometricFeaturesV2(np.ndarray[FLOAT_t, ndim=2] frames, list joints, list ignored_joints, list ignored_joints_index):
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

    cdef np.ndarray[FLOAT_t, ndim=3] coords_frames = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)
    cdef float [:,:,:] coords_frames_view = coords_frames
    # For each frame, compute all JJD (Joint-Joint Distance) and JJO (Joint-Joint Orientation) for Pose Features (PF) and Motion Features (MF)
    D_original_pose = []
    O_original_pose = []
    D_original_motion = []
    O_original_motion = []

    cdef:
        np.ndarray frame_t, next_frame, orientation_motion, orientation_pose, tmp_j, tmp_k, tmp_j_1, tmp_k_1
        np.ndarray[FLOAT_t, ndim=1] JJD_pose, JJD_motion
        np.ndarray[FLOAT_t, ndim=2] JJO_pose, JJO_motion
        np.ndarray[FLOAT_t, ndim=1] root_t
        float [:] JJD_pose_view
        float [:] JJD_motion_view
        float [:,:] JJO_pose_view
        float [:,:] JJO_motion_view
        int t, joint_J, joint_K, i
        int N=coords_frames.shape[0]
        int M=coords_frames.shape[1]
        float euclidean_distance_motion, euclidean_distance_pose

    for t in xrange(N):
        frame_t = coords_frames[t]
        if t != N-1:
            next_frame = coords_frames[t+1]
        JJD_pose = np.empty((20), dtype=FLOAT)
        JJD_motion = np.empty((20), dtype=FLOAT)
        JJO_pose = np.empty((20, 3), dtype=FLOAT)
        JJO_motion = np.empty((20, 3), dtype=FLOAT)
        JJD_pose_view = JJD_pose
        JJD_motion_view = JJD_motion
        JJO_pose_view = JJO_pose
        JJO_motion_view = JJO_motion
        root_t = frame_t[0]
        i = 0
        for joint in xrange(1, M):
            tmp_j = frame_t[joint]

            euclidean_distance_pose = np.linalg.norm(tmp_j - root_t)
            JJD_pose[i]= euclidean_distance_pose
            if t != N-1:
                tmp_j_1 = next_frame[joint]
                euclidean_distance_motion = np.linalg.norm(tmp_j_1 - root_t)
                JJD_motion[i] = euclidean_distance_motion
            
            orientation_pose = root_t - tmp_j
            JJO_pose[i] = orientation_pose

            if t != N-1:
                tmp_j_1 = next_frame[joint]
                orientation_motion = root_t - tmp_j_1
                JJO_motion[i] = orientation_motion

            i+=1

        D_original_pose.append(JJD_pose)
        O_original_pose.append(JJO_pose)
        
        if t != N-1: 
            D_original_motion.append(JJD_motion)
            O_original_motion.append(JJO_motion)

    D_original_pose = np.array(D_original_pose)
    D_original_motion = np.array(D_original_motion)

    # Normalize D_original_pose and D_original_motion between [0; 255] and apply JET color map on them to obtain two images with 3 channels (R, G, B)
    cdef int D_min_pose = 0
    cdef int D_min_motion = 0
    
    D_max_pose = np.amax(D_original_pose)
    D_max_motion = np.amax(D_original_motion)

    cdef np.ndarray D_norm_pose = 255 * ((D_original_pose - D_min_pose) / (D_max_pose - D_min_pose))
    D_norm_pose = np.array(D_norm_pose, dtype=UINT)
    
    cdef np.ndarray D_norm_motion = 255 * ((D_original_motion - D_min_motion) / (D_max_motion - D_min_motion))
    D_norm_motion = np.array(D_norm_motion, dtype=UINT)

    cdef np.ndarray JJD_RGB_pose = cv2.applyColorMap(D_norm_pose, cv2.COLORMAP_JET)
    cdef np.ndarray JJD_RGB_motion = cv2.applyColorMap(D_norm_motion, cv2.COLORMAP_JET)

    O_original_pose = np.array(O_original_pose)
    O_original_motion = np.array(O_original_motion)

    # Normalize between [0; 255] and consider (x, y, z) as (R, G, B)
    c_max_pose = np.amax(O_original_pose)
    c_max_motion = np.amax(O_original_motion)

    c_min_pose = np.amin(O_original_pose)
    c_min_motion = np.amin(O_original_motion)

    JJO_RGB_pose = np.array(np.floor(255 * ((O_original_pose - c_min_pose) / (c_max_pose - c_min_pose))), dtype=UINT)
    JJO_RGB_motion = np.array(np.floor(255 * ((O_original_motion - c_min_motion) / (c_max_motion - c_min_motion))), dtype=UINT)

    cdef np.ndarray[UINT_t, ndim=3] PF = np.empty((len(coords_frames), 40,  3), dtype=UINT)

    cdef np.ndarray[UINT_t, ndim=3] MF = np.empty((len(coords_frames)-1, 40, 3), dtype=UINT)

    for nb_frame in xrange(len(coords_frames)):
        temp_pf = np.concatenate([JJD_RGB_pose[nb_frame], JJO_RGB_pose[nb_frame]])
        PF[nb_frame] = temp_pf
        if nb_frame<len(coords_frames)-1:
            temp_mf = np.concatenate([JJD_RGB_motion[nb_frame], JJO_RGB_motion[nb_frame]])
            MF[nb_frame] = temp_mf

    cdef np.ndarray[UINT_t, ndim=3] SPMF = np.empty((40, 2*len(coords_frames)-1, 3), dtype=UINT)

    cdef int frame_index = 0
    cdef np.ndarray pf, mf
    for pf, mf in zip(PF, MF):
        SPMF[:, frame_index] = pf
        frame_index += 1
        SPMF[:, frame_index] = mf
        frame_index += 1

    # Add the last element of PF to SPMF because the zip function keeps only the elements in common between the two tables. 
    # The PF table has one element more than MF.
    SPMF[:, frame_index] = PF[len(PF)-1]

    return SPMF

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[UINT_t, ndim=3] bvh2GeometricFeaturesCustom(np.ndarray[FLOAT_t, ndim=2] frames, list joints, list ignored_joints, list ignored_joints_index, np.ndarray[UINT_t, ndim=1] joints_we_want):
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

    cdef np.ndarray[FLOAT_t, ndim=3] coords_frames = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)
    cdef float [:,:,:] coords_frames_view = coords_frames
    # For each frame, compute all JJD (Joint-Joint Distance) and JJO (Joint-Joint Orientation) for Pose Features (PF) and Motion Features (MF)
    D_original_pose = []
    O_original_pose = []
    D_original_motion = []
    O_original_motion = []

    cdef:
        np.ndarray frame_t, next_frame, orientation_motion, orientation_pose, tmp_j, tmp_k, tmp_j_1, tmp_k_1
        np.ndarray[FLOAT_t, ndim=1] JJD_pose, JJD_motion
        np.ndarray[FLOAT_t, ndim=2] JJO_pose, JJO_motion
        list references
        int t, joint_J, joint_K, i
        int N=coords_frames.shape[0]
        int M=coords_frames.shape[1]
        float euclidean_distance_motion, euclidean_distance_pose

    for t in xrange(N):
        frame_t = coords_frames[t]
        if t != N-1:
            next_frame = coords_frames[t+1]

        JJD_pose = np.empty((100), dtype=FLOAT)
        JJD_motion = np.empty((100), dtype=FLOAT)
        JJO_pose = np.empty((100, 3), dtype=FLOAT)
        JJO_motion = np.empty((100, 3), dtype=FLOAT)

        references = [frame_t[j] for j in joints_we_want]
        
        i = 0
        
        for joint_J in xrange(M):
            for joint_K, ref in zip(joints_we_want, references):
                if joint_J == joint_K:
                    continue
                
                tmp_j = frame_t[joint_J]

                
                JJD_pose[i] = np.linalg.norm(tmp_j - ref)
                JJO_pose[i] = ref - tmp_j
                if t != N-1:
                    tmp_j_1 = next_frame[joint_J]
                    JJD_motion[i] = np.linalg.norm(tmp_j_1 - ref)
                    JJO_motion[i] = ref - tmp_j_1

                i+=1

        D_original_pose.append(JJD_pose)
        O_original_pose.append(JJO_pose)
        
        if t != N-1: 
            D_original_motion.append(JJD_motion)
            O_original_motion.append(JJO_motion)

    D_original_pose = np.array(D_original_pose)
    D_original_motion = np.array(D_original_motion)

    # Normalize D_original_pose and D_original_motion between [0; 255] and apply JET color map on them to obtain two images with 3 channels (R, G, B)
    cdef int D_min_pose = 0
    cdef int D_min_motion = 0
    
    D_max_pose = np.amax(D_original_pose)
    D_max_motion = np.amax(D_original_motion)

    cdef np.ndarray D_norm_pose = 255 * ((D_original_pose - D_min_pose) / (D_max_pose - D_min_pose))
    D_norm_pose = np.array(D_norm_pose, dtype=UINT)
    
    cdef np.ndarray D_norm_motion = 255 * ((D_original_motion - D_min_motion) / (D_max_motion - D_min_motion))
    D_norm_motion = np.array(D_norm_motion, dtype=UINT)

    cdef np.ndarray JJD_RGB_pose = cv2.applyColorMap(D_norm_pose, cv2.COLORMAP_JET)
    cdef np.ndarray JJD_RGB_motion = cv2.applyColorMap(D_norm_motion, cv2.COLORMAP_JET)

    O_original_pose = np.array(O_original_pose)
    O_original_motion = np.array(O_original_motion)

    # Normalize between [0; 255] and consider (x, y, z) as (R, G, B)
    c_max_pose = np.amax(O_original_pose)
    c_max_motion = np.amax(O_original_motion)

    c_min_pose = np.amin(O_original_pose)
    c_min_motion = np.amin(O_original_motion)

    JJO_RGB_pose = np.array(np.floor(255 * ((O_original_pose - c_min_pose) / (c_max_pose - c_min_pose))), dtype=UINT)
    JJO_RGB_motion = np.array(np.floor(255 * ((O_original_motion - c_min_motion) / (c_max_motion - c_min_motion))), dtype=UINT)

    cdef np.ndarray[UINT_t, ndim=3] PF = np.empty((len(coords_frames), 200,  3), dtype=UINT)

    cdef np.ndarray[UINT_t, ndim=3] MF = np.empty((len(coords_frames)-1, 200, 3), dtype=UINT)

    for nb_frame in xrange(len(coords_frames)):
        temp_pf = np.concatenate([JJD_RGB_pose[nb_frame], JJO_RGB_pose[nb_frame]])
        PF[nb_frame] = temp_pf
        if nb_frame<len(coords_frames)-1:
            temp_mf = np.concatenate([JJD_RGB_motion[nb_frame], JJO_RGB_motion[nb_frame]])
            MF[nb_frame] = temp_mf

    cdef np.ndarray[UINT_t, ndim=3] SPMF = np.empty((200, 2*len(coords_frames)-1, 3), dtype=UINT)

    cdef int frame_index = 0
    cdef np.ndarray pf, mf
    for pf, mf in zip(PF, MF):
        SPMF[:, frame_index] = pf
        frame_index += 1
        SPMF[:, frame_index] = mf
        frame_index += 1

    # Add the last element of PF to SPMF because the zip function keeps only the elements in common between the two tables. 
    # The PF table has one element more than MF.
    SPMF[:, frame_index] = PF[len(PF)-1]

    return SPMF

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list bvh2MultipleImages(np.ndarray[FLOAT_t, ndim=2] frames, list joints, list ignored_joints, list ignored_joints_index):

    cdef np.ndarray[FLOAT_t, ndim=3] world_coordinates = getWorldCoordinates(frames, joints, ignored_joints, ignored_joints_index)
    
    cdef:
        int F = len(world_coordinates)
        int N = len(world_coordinates[0])
        int f, n
        float energy
        np.ndarray[FLOAT_t, ndim=2] current_frame
        np.ndarray[FLOAT_t, ndim=2] next_frame
        list energies_list = [0 for i in range(N)]
        int step = 10
    
    for f in range(0, F-step, step):
        current_frame = world_coordinates[f]
        next_frame = world_coordinates[f+step]
        for n in range(N):
            energy = np.linalg.norm(next_frame[n] - current_frame[n])
            energies_list[n] += energy

    cdef np.ndarray energies = np.array(energies_list)
    energies = (energies - np.amin(energies))/(np.amax(energies) - np.amin(energies))
    
    cdef:
        float rho = .8

        np.ndarray weights = rho*energies + (1 - rho)

        list images = [[[], [] ,[]] for i in range(10)]
        float j, k, r, g, b
        np.ndarray rgb
    
    for f in range(F):
        for n in range(N):
            coords_5D = np.concatenate((world_coordinates[f][n], np.array([f, n])))
            for c in range(3):
                current_2D = space_2D_v2[c]
                current_3D = space_3D_v2[c]
                j = coords_5D[current_2D[0]]
                k = coords_5D[current_2D[1]]
                r = coords_5D[current_3D[0]]
                g = coords_5D[current_3D[1]]
                b = coords_5D[current_3D[2]]
                rgb = np.array([r, g, b])
                rgb = (1 - weights[n])*np.array([255, 255, 255]) + weights[n]*rgb
                images[c][0].append(j)
                images[c][1].append(k)
                images[c][2].append(rgb)
    
    cdef:
        float beta = .8
        int img_index
        np.ndarray X, Y, RGB
        list final_images = []

    for img_index in range(3):
        X = np.array(images[img_index][1], dtype=FLOAT)
        Y = np.array(images[img_index][0], dtype=FLOAT)
        RGB = np.array(images[img_index][2])
        RGB = np.floor(255 * (RGB - np.amin(RGB))/(np.amax(RGB) - np.amin(RGB)))

        x_min, x_max = np.amin(X), np.amax(X)
        y_min, y_max = np.amin(Y), np.amax(Y)

        temp_w = x_max - x_min
        temp_h = y_max - y_min

        if temp_w > temp_h:
            mult = (temp_w // temp_h)
            Y = Y * mult
        else:
            mult = (temp_h // temp_w)
            X = X * mult
        
        X = np.array(np.floor(X*10), dtype=INT)
        Y = np.array(np.floor(Y*10), dtype=INT)

        x_min, x_max = np.amin(X)-10, np.amax(X)+10
        y_min, y_max = np.amin(Y)-10, np.amax(Y)+10

        width, height = x_max - x_min, y_max - y_min

        img = np.zeros((height, width, 3), dtype=UINT)
        img.fill(255)
        for x, y, rgb in zip(X, Y, RGB):
            if x_max <= width:
                x += np.abs(x_min)
            if x_max >= width:
                x -= np.abs(x_min)
            if y_max >= height:
                y -= np.abs(y_min)
            if y_max <= height:
                y += np.abs(y_min)

            cv2.circle(img, (x, height-y), int((1-beta)*10+beta*(width/100)), [int(rgb[2]), int(rgb[1]), int(rgb[0])], -1)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        final_images.append(img)
    return final_images