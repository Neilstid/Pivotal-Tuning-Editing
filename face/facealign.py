# !/usr/bin/env python
# coding=utf8

import os
import math
from typing import NoReturn, List
import cv2
import numpy as np
import dlib
from scipy.spatial import Delaunay # pylint: disable=E0611


#==============================================================================
# Constants
#==============================================================================
# Landmarks constants
LANDMARKS_DLIB = 0
LANDMARKS_MEDIAPIPE = 1
LANDMARKS_FACE2POINTS = 2
LANDMARKS_DLIB_FACE2POINTS = 3
LANDMARKS_MEDIAPIPE_FACE2POINTS = 4


#==============================================================================
# Module initialisation
#==============================================================================

# Get the dlib model
DLIB_DATA_DIR: str = os.environ.get(
    'DLIB_DATA_DIR',
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dlib')
)
dlib_detector = dlib.get_frontal_face_detector() # pylint: disable=E1101
dlib_predictor = dlib.shape_predictor( # pylint: disable=E1101
    os.path.join(DLIB_DATA_DIR, 'shape_predictor_68_face_landmarks.dat')
)


def align(
    images: list, all_points: list, landmark_model: int = LANDMARKS_DLIB,
    skip_ref: bool = True
) -> List[np.array]:


    # Corners of the eye in input image
    if landmark_model == LANDMARKS_DLIB or landmark_model == LANDMARKS_DLIB_FACE2POINTS:
        eyecorner_dst: list = [all_points[0][36], all_points[0][45]]
    elif landmark_model == LANDMARKS_FACE2POINTS:
        eyecorner_dst: list = [all_points[0][0], all_points[0][51]]
    elif landmark_model == LANDMARKS_MEDIAPIPE or landmark_model == LANDMARKS_MEDIAPIPE_FACE2POINTS:
        eyecorner_dst: list = [all_points[0][129], all_points[0][358]]
    else:
        eyecorner_dst: list = all_points[0]

    # Eye corners
    h, w, _ = images[0].shape

    images_norm: list = []
    points_norm: list = []

    # Add boundary points for delaunay triangulation
    boundary_pts: np.array = np.array([
        (0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2),
        (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)
    ])

    # Initialize location of average points to 0s
    points_avg: np.array = np.array(
        [(0, 0)] * (len(all_points[0]) + len(boundary_pts)),
        np.float32
    )

    # Get the number of image
    num_images: int = len(images)

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    for i in range(0, num_images):

        points1: np.array = all_points[i]

        # Corners of the eye in input image
        if landmark_model == LANDMARKS_DLIB or landmark_model == LANDMARKS_DLIB_FACE2POINTS:
            eyecorner_src: list = [all_points[i][36], all_points[i][45]]
        elif landmark_model == LANDMARKS_FACE2POINTS:
            eyecorner_src: list = [all_points[i][0], all_points[i][51]]
        elif landmark_model == LANDMARKS_MEDIAPIPE or landmark_model == LANDMARKS_MEDIAPIPE_FACE2POINTS:
            eyecorner_src: list = [all_points[i][129], all_points[i][358]]
        else:
            eyecorner_src: list = all_points[i]

        # Compute similarity transform
        tform: list = similarity_transform(eyecorner_src, eyecorner_dst)

        # Apply similarity transformation
        img: np.array = cv2.warpAffine(images[i], tform, (w, h)) # pylint: disable=E1101

        # Apply similarity transform on points
        points2: np.array = np.reshape(np.array(points1), (points1.shape[0], 1, 2))
        points: np.array = cv2.transform(points2, tform) # pylint: disable=E1101
        points: np.array = np.float32(np.reshape(points, (points1.shape[0], 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points: np.array = np.append(points, boundary_pts, axis=0)

        # Calculate location of average landmark points.
        if i == 0:
            points_avg: np.array = points

        points_norm.append(points)
        images_norm.append(img)

    # Delaunay triangulation
    rect: tuple = (0, 0, w, h)
    tri = calculate_triangles(np.array(points_avg))

    # Output image
    output = []

    # Warp input images to average image landmarks
    for i in range(1 if skip_ref else 0, num_images):
        img: np.array = np.zeros((h, w, 3), np.float32)
        # Transform triangles one by one
        for j in range(0, len(tri)):
            t_in = []
            t_out = []

            for k in range(0, 3):
                p_in = points_norm[i][tri[j][k]]
                p_in = constrain_point(p_in, w, h)

                p_out = points_avg[tri[j][k]]
                p_out = constrain_point(p_out, w, h)

                t_in.append(p_in)
                t_out.append(p_out)

            warp_triangle(images_norm[i], img, t_in, t_out)

        # Add image intensities for averaging
        img = img.astype(np.uint8)
        output.append(img)

    return output


# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarity_transform(in_points: list, out_points: list) -> list:
    """
    Compute similarity transform given two sets of two points

    :param in_points: 
    :type in_points: list
    :param out_points: 
    :type out_points: list
    :return: 
    :rtype: list
    """
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    in_pts = np.copy(in_points).tolist()
    out_pts = np.copy(out_points).tolist()

    xin = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * \
        (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * \
        (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

    in_pts.append([np.int32(xin), np.int32(yin)])

    xout = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * \
        (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * \
    (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    out_pts.append([np.int32(xout), np.int32(yout)])

    tform = cv2.estimateAffinePartial2D(np.array([in_pts]), np.array([out_pts])) # pylint: disable=E1101
    
    return tform[0]

# Check if a point is inside a rectangle
def rect_contains(rect: list, point: list) -> bool:
    """
    Check if a point is inside a rectangle

    :param rect: Rectangle
    :type rect: list
    :param point: Point
    :type point: list
    :return: True if point is in rectangle
    :rtype: bool
    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def calculate_triangles(points: list) -> list:
    return np.array([(p[0], p[1], p[2]) for p in Delaunay(points).simplices])


def constrain_point(p: list, w: int, h: int) -> tuple:
    # Check that every points is in the image
    return [min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1)]


# Apply affine transform calculated using src_tri and dst_tri to src and
# output an image of size.
def apply_affine_transform(src: np.array, src_tri: np.array, dst_tri: np.array, size: list) -> np.array:

    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri)) # pylint: disable=E1101

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, # pylint: disable=E1101
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) # pylint: disable=E1101

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warp_triangle(
    img1: np.array, img2: np.array, t1: np.array, t2: np.array
) -> NoReturn:
    """
    Wrap the triangle

    :param img1: First image
    :type img1: np.array
    :param img2: Second image
    :type img2: np.array
    :param t1: Triangle of img1
    :type t1: np.array
    :param t2: Triangle of img2
    :type t2: np.array
    :return: Nothing
    :rtype: NoReturn
    """

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1])) # pylint: disable=E1101
    r2 = cv2.boundingRect(np.float32([t2])) # pylint: disable=E1101

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0) # pylint: disable=E1101

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

