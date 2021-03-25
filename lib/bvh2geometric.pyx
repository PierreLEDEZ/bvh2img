# distutils: language=c++
# cython: cdivision=True
# cython: language_level=3

import numpy as np, cv2, time
cimport numpy as np
cimport cython

UINT=np.uint8
FLOAT=np.float32

ctypedef np.uint8_t UINT_t
ctypedef np.float32_t FLOAT_t

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
        indexes = [[joint_J, joint_K] for joint_J in xrange(M-1) for joint_K in xrange(joint_J+1, M)]
        for joint_J, joint_K in indexes:
        
            tmp_j = frame_t[joint_J]
            tmp_k = frame_t[joint_K]

            euclidean_distance_pose = np.linalg.norm(tmp_j - tmp_k)
            # JJD_pose.append(euclidean_distance_pose)
            JJD_pose[i] = euclidean_distance_pose
            if t != N-1:
                tmp_k_1 = next_frame[joint_K]
                euclidean_distance_motion = np.linalg.norm(tmp_j - tmp_k_1)
                JJD_motion[i] = euclidean_distance_motion
                # JJD_motion.append(euclidean_distance_motion)
            
            orientation_pose = tmp_k - tmp_j

            # JJO_pose.append(orientation_pose)
            JJO_pose[i] = orientation_pose

            if t != N-1:
                tmp_j_1 = next_frame[joint_J]
                orientation_motion = tmp_k - tmp_j_1
                # JJO_motion.append(orientation_motion)
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