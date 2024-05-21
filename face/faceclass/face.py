# Python Basic Libraries
from __future__ import annotations
from typing import List, Tuple, Union, Any
from os.path import basename, dirname, realpath, join as path_join, exists as path_exists
import os
from uuid import uuid4
from copy import copy, deepcopy
import torch

# Python Libraries
from cv2 import imread, cvtColor, COLOR_BGR2RGB, resize, imwrite, imshow,\
    waitKey, circle, putText, FONT_HERSHEY_SIMPLEX, addWeighted, fillPoly, error
import cv2
import dlib
import mediapipe as mp
from numpy import array as np_array, int32, zeros as np_zeros, save as np_save,\
    max as np_max, uint8, where as np_where, min as np_min, abs as np_abs,any as np_any
import numpy as np
from numpy.linalg import norm
import imutils
from face.facealign import align
import warnings as warn
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion


# Modules
from face.resnet_mask import FaceParser


#==============================================================================
# Constants
#==============================================================================
# Landmarks constants
LANDMARKS_DLIB = 0
LANDMARKS_MEDIAPIPE = 1


# Path to the semantic folder as default
SEMANTIC_FOLDER_NAME: str = "semantic"
LANDMARKS_FOLDER_NAME: str = "landmarks"

USED_COLOR_OVERLAY: np_array = np_array([
    [255, 35, 35],
    [187, 210, 14],
    [128, 102, 164],
    [163, 192, 204],
    [56, 72, 35],
    [79, 58, 43],
    [113, 75, 84],
    [180, 146, 132],
    [245, 16, 44],
    [223, 138, 174],
    [13, 120, 124],
    [153, 202, 233],
    [6, 143, 196],
    [6, 196, 181],
    [6, 196, 98],
    [33, 5, 196],
    [134, 6, 196],
    [196, 6, 147],
    [196, 6, 6],
    [54, 248, 3],
    [242, 167, 5]
])

FACE_MESH: mp.solutions.Solution = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)


LANDMARKS_REGION = {
    "LionWrinkle": [107, 108, 151, 337, 336, 285, 413, 168, 189, 55],
    "LionWrinkleUp": [107, 108, 151, 337, 336, 285, 8, 55],
    "UpperLipWrinkle": [40, 92, 165, 167, 164, 393, 394, 322, 270, 269, 267, 0, 37, 39],
    "Forehead": [70, 71, 21, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 251, 301, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63],
    "MarionetteWrinkle": [
        [409, 410, 436, 434, 364, 365, 379, 378, 395, 431, 424, 335, 321, 375, 291], 
        [185, 186, 216, 214, 135, 136, 150, 149, 170, 211, 204, 106, 91, 146, 61]
    ],
    "UnderEyeWrinkle": [
        [463, 341, 256, 252, 253, 254, 339, 255, 261, 340, 345, 352, 280, 330, 329, 277, 343, 412, 465, 464],
        [243, 112, 26, 22, 23, 24, 110, 25, 31, 111, 116, 123, 50, 101, 100, 47, 114, 188, 245, 244]
    ],
    "NasobialWrinkle": [
        [358, 266, 425, 427, 434, 432, 287, 410, 322, 391, 327],
        [129, 36, 205, 207, 214, 212, 57, 186, 92, 165, 98]
    ],
    "Lips": [
        57, 185, 40, 39, 37, 0, 267, 269, 270, 409, 287, 375, 321, 405, 314, 17, 84, 181, 91, 146
    ]
}

MASK_VALUE = {
    "Lips": [12, 13],
    "Hair": [17]
}


#==============================================================================
# Module initialisation 
#==============================================================================


# Get the dlib model
DLIB_DATA_DIR: str = os.environ.get(
    'DLIB_DATA_DIR',
    path_join(dirname(dirname(realpath(__file__))), 'dlib')
)
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(
    path_join(DLIB_DATA_DIR, 'shape_predictor_68_face_landmarks.dat')
)


#==============================================================================
# Class
#==============================================================================
# @propagate_decorator(check_types(level=LOG))
class Face:
    """
    Tool box for face processing
    """

    # Face parser used to determine the semantic of the face
    face_parser = FaceParser()

    # Class attributes
    def __init__(
        self, path: str, image: np_array = None, semantic: np_array = None
    ) -> None:
        """
        Constructor

        :param path: Path to the image
        :type path: str
        :param image: Image (in BGR format), defaults to None
        :type image: np_array, optional
        :param semantic: Semantic of the image, defaults to None
        :type semantic: np_array, optional
        :return: Nothing it's a constructor ;)
        :rtype: None
        """
        self.path: str = path
        self.image: np_array = image
        self.semantic: np_array = semantic
        self.landmark_cache = dict()

    
    def __copy__(self) -> Face:
        """
        Copy the face

        :return: A copy of the face
        :rtype: Face
        """
        return Face(self.path, self.image, self.semantic)


    def read_image(self, path: str = None) -> Face:
        """
        Method to read the image

        :param path: _description_, defaults to None
        :type path: str, optional
        :return: _description_
        :rtype: Face

        ..warning:: This should be called only if you are using a path to an image and not the image itself.
        """
        if path is None:
            path = self.path

        self.image = imread(path)
        self.landmark_cache = dict()

        return self


    def build(self, semantic_path: str = None) -> None:
        """
        Read the image and the semantic based on the path

        :param semantic_path: Path to the semantic data, defaults to None
        :type semantic_path: str, optional
        :return: Nothing
        :rtype: None
        """
        # If the semantic has not been precised
        if semantic_path is None:
            # Determine the semantic folder based on the image folder
            semantic_path: str = path_join(dirname(dirname(self.path)), SEMANTIC_FOLDER_NAME, basename(self.path))

        # Read the image and the semantic
        self.image = imread(self.path)
        self.semantic = imread(self.semantic_path)

        # Assert they have the same size
        if self.image.shape[:2] != self.semantic.shape[:2]:
            resize(self.semantic, self.image.shape[:2])


    def landmark(self, landmark_model: int = LANDMARKS_MEDIAPIPE, rects=None) -> np_array:
        """
        Compute the landmark of the face

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :return: Landmarks of the face
        :rtype: np_array
        """

        try:
            return self.landmark_cache[landmark_model]
        except KeyError:
            pass

        # For dlib landmarks
        if landmark_model == LANDMARKS_DLIB:
            # Convert the image to rgb
            rgbimg: np_array = cvtColor(self.image, COLOR_BGR2RGB)
            rects: np_array = dlib_detector(rgbimg, 1)

            # If no face found raise an error
            if len(rects) == 0:
                raise RuntimeError("No face found !")

            # Determine the landmarks
            shapes = dlib_predictor(rgbimg, rects[0])
            # Parse the points
            points: np_array = np_array([(shapes.part(i).x, shapes.part(i).y) for i in range(68)], int32)
        # For mediapipe landmarks
        elif landmark_model == LANDMARKS_MEDIAPIPE:
            # Get the size of the image (it will be usefull for the landmarks points since they are normalized)
            width, height = self.size
            # Create the model
            # Get the mesh of the face
            mesh: mp.solutions.Solution = FACE_MESH.process(cvtColor(self.image, COLOR_BGR2RGB))
            # Parse the points
            points: np_array = np_array([
                (int(lm.x * height), int(lm.y * width)) for lm in mesh.multi_face_landmarks[0].landmark
            ], int32)
 
        # Save the landmark for a quick loading later
        self.landmark_cache[landmark_model] = points

        return points


    def face_crop(
        self, landmark_model: int = LANDMARKS_MEDIAPIPE, auto_update: bool = True
    ) -> Tuple[np_array, np_array]:
        """
        Crop over a single face

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :param auto_update: True to update the class, False either, defaults to True
        :type auto_update: bool, optional
        :return: Crop over a face in the image and in the semantic
        :rtype: Tuple[np_array, np_array]
        """

        # Clear the landmark cache since image will be update
        self.landmark_cache = dict()

        width, heigh = self.size
        # Get the landmark
        landmarks: np_array = self.landmark(landmark_model)

        # Transpose the array to have seperatly the x in array and the y in an other one
        landmarks_T: np_array = landmarks.T

        # Determine the limits of landmarks
        max_x, max_y = max(landmarks_T[0]), max(landmarks_T[1])
        min_x, min_y = min(landmarks_T[0]), min(landmarks_T[1])

        # Build a rectantangle over the landmarks and crop over
        crop_image: np_array = self.image[
            int(max(0, min_y * 0.9)):int(min(width, max_y * 1.1)), 
            int(max(0, min_x * 0.9)):int(min(heigh, max_x * 1.25))
        ]
        crop_semantic: np_array = self.image[
            int(max(0, min_y * 0.9)):int(min(width, max_y * 1.1)), 
            int(max(0, min_x * 0.9)):int(min(heigh, max_x * 1.25))
        ]

        # If the autoupdate is True, update the value of its own attributes
        if auto_update:
            self.image = crop_image
            self.semantic = crop_semantic

        return crop_image, crop_semantic


    def clear_landmarks(self) -> None:
        """
        Clear the cache for the computed landmarks
        """
        self.landmark_cache = dict()


    def square_crop(
        self, landmark_model: int = LANDMARKS_DLIB, auto_update: bool = True,
        return_quad: bool = False
    ) -> Tuple[np_array, np_array]:
        """
        Create a crop arround a single face in a square shape

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :param auto_update: True to update the class, False otherwise, defaults to True
        :type auto_update: bool, optional
        :raises NotImplementedError: An impossible case
        :return: Crop over a face in the image and in the semantic
        :rtype: Tuple[np_array, np_array]
        """

        # Clear the landmark cache since image will be update
        self.landmark_cache = dict()
        # Get the size of the image
        width, heigh = self.size
        # Get the landmark
        landmarks: np_array = self.landmark(landmark_model)

        # Transpose the array to have seperatly the x in array and the y in an other one
        landmarks_T: np_array = landmarks.T

        # Determine the limits of landmarks
        max_x = min(heigh, max(landmarks_T[0]) + ((max(landmarks_T[0]) - min(landmarks_T[0])) * 0.05)) 
        max_y = min(width, max(landmarks_T[1]) + ((max(landmarks_T[1]) - min(landmarks_T[1])) * 0.1))
        min_x = max(0, min(landmarks_T[0]) - ((max(landmarks_T[0]) - min(landmarks_T[0])) * 0.05))
        min_y = max(0, min(landmarks_T[1]) - ((max(landmarks_T[1]) - min(landmarks_T[1])) * 0.1))

        # Compute the size of the crop
        nb_px_x: float = max_x - min_x
        nb_px_y: float = max_y - min_y

        # (y, width)
        # ^
        # |
        # .___> (x, heigh)
        #
        if nb_px_x == nb_px_y:
            pass
        elif nb_px_x > nb_px_y and nb_px_x <= width:
            # Compute the adding to min_x and max_x
            decrease_min_y: int = min((nb_px_x - nb_px_y) // 2, min_y)
            increase_max_y: int = min((nb_px_x - nb_px_y) - decrease_min_y, width - max_y)
            # Add the rest to min_x
            decrease_min_y += ((nb_px_x - nb_px_y) - decrease_min_y) - increase_max_y
            # Update the value y
            min_y, max_y = min_y - decrease_min_y, max_y + increase_max_y
        elif nb_px_y > nb_px_x and nb_px_y <= heigh:
            # Compute the adding to min_y and max_y
            decrease_min_x: int = min((nb_px_y - nb_px_x) // 2, min_x)
            increase_max_x: int = min((nb_px_y - nb_px_x) - decrease_min_x, heigh - max_x)
            # Add the rest to min_y
            decrease_min_x += ((nb_px_y - nb_px_x) - decrease_min_x) - increase_max_x
            # Update the value x
            min_x, max_x = min_x - decrease_min_x, max_x + increase_max_x
        elif nb_px_x > nb_px_y:
            # Increase at the maximum possible to loose the min of information
            min_y, max_y = 0, width
            # Update the nb_px_y
            nb_px_y = width

            # Compute the adding to min_y and max_y
            increase_min_x: int = min((nb_px_x - nb_px_y) // 2, min_x)
            decrease_max_x: int = min((nb_px_x - nb_px_y) - increase_min_x, heigh - max_x)
            # Add the rest to min_y
            increase_min_x += ((nb_px_x - nb_px_y) - increase_min_x) - decrease_max_x
            # Update the value x
            min_x, max_x = min_x + increase_min_x, max_x - decrease_max_x
        elif nb_px_y > nb_px_x:
            # Increase at the maximum possible to loose the min of information
            min_x, max_x = 0, heigh
            # Update the nb_px_y
            nb_px_x = heigh

            # Compute the adding to min_y and max_y
            increase_min_y: int = min((nb_px_y - nb_px_x) // 2, min_y)
            decrease_max_y: int = min((nb_px_y - nb_px_x) - increase_min_y, width - max_y)
            # Add the rest to min_y
            increase_min_y += ((nb_px_y - nb_px_x) - increase_min_y) - decrease_max_y
            # Update the value x
            min_y, max_y = min_y + increase_min_y, max_y - decrease_max_y
        else:
            raise NotImplementedError("Hey where are we ? Is it even possible ?") # ;)

        # Build a rectantangle over the landmarks and crop over
        crop_image: np_array = self.image[int(min_y):int(max_y), int(min_x):int(max_x)]
        try:
            crop_semantic: np_array = self.semantic[int(min_y):int(max_y), int(min_x):int(max_x)]
        except:
            warn.warn("No semantic found. If one, it won't be resized!")
            crop_semantic: np_array = np_array([])

        # If the autoupdate is True, update the value of its own attributes
        if auto_update:
            self.image = crop_image
            try:
                self.semantic = crop_semantic
            except:
                pass

        if return_quad:
            return [int(min_y), int(max_y), int(min_x), int(max_x)]

        return crop_image, crop_semantic


    @property
    def name(self) -> str:
        """
        Name getter

        :return: The name of the image
        :rtype: str
        """
        return basename(self.path).split(".")[0]


    @property
    def size(self) -> Tuple[int, int]:
        """
        Size getter

        :return: The shape of the image
        :rtype: Tuple[int, int]
        """
        return self.image.shape[:2]


    def resize(self, size: Tuple[int, int]) -> None:
        """
        Resize the image and the semantic

        :param size: New size of image and semantic
        :type size: Tuple[int, int]
        :return: Nothing it just update itself :p
        :rtype: None
        """
        # Clear the landmark cache since image will be update
        self.landmark_cache = dict()

        self.image = resize(self.image, size)

        try:
            self.semantic = resize(self.semantic, size, interpolation=cv2.INTER_NEAREST_EXACT)
        except error:
            warn.warn("Semantic is null and cannot be resize")

        return


    def save(
        self, image_path: str = None, semantic_path: str = None, landmarks_path: str = None,
        overwrite: bool = True, save_landmarks: bool = False,
        save_semantic: bool = False,
        landmark_model: int = LANDMARKS_MEDIAPIPE
    ) -> None:
        """
        Save the image, semantic and landmarks

        :param semantic_path: Path to the semantic file, defaults to None
        :type semantic_path: str, optional
        :param landmarks_path: Path to the landmark file, defaults to None
        :type landmarks_path: str, optional
        :param overwrite: True if it can overwrite the previous file, else False, defaults to True
        :type overwrite: bool, optional
        :param save_landmarks: True if the landmarks have to be saved too, False otherwise, defaults to True
        :type save_landmarks: bool, optional
        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :return: Nothing
        :rtype: None
        """

        # If the save can overwrite the images
        if overwrite:

            # If the semantic is not define
            if semantic_path is None:
                # Determine the semantic folder based on the image folder
                semantic_path: str = path_join(
                    dirname(dirname(self.path)), SEMANTIC_FOLDER_NAME,
                    basename(self.path)
                )

            # If the landmarks path is not define
            if landmarks_path is None:
                # Determine the landmarks folder based on the image folder
                landmarks_path: str = path_join(
                    dirname(dirname(self.path)), LANDMARKS_FOLDER_NAME,
                    basename(self.path).split(".")[0] + ".npy"
                )

            if os.path.splitext(self.path)[1] != os.path.splitext(image_path)[1]:
                os.remove(self.path)
                self.path = os.path.splitext(self.path)[0] + os.path.splitext(image_path)[1]

        # If the semantic has not been precised
        elif not overwrite and image_path is None:
            # Create an unique name
            img_name: str = str(uuid4())
            # Update the path
            self.path = path_join(dirname(self.path), f"{img_name}.png")
            # Determine the semantic folder based on the image folder
            semantic_path: str = path_join(
                dirname(dirname(self.path)), SEMANTIC_FOLDER_NAME, f"{img_name}.png"
            )
            # Determine the landmarks folder based on the image folder
            landmarks_path: str = path_join(
                dirname(dirname(self.path)), LANDMARKS_FOLDER_NAME, f"{img_name}.npy"
            )
        else:
            # Update the path
            self.path = image_path
            img_name: str = Path(self.path).stem
            # Determine the semantic folder based on the image folder
            semantic_path: str = path_join(
                dirname(dirname(self.path)), SEMANTIC_FOLDER_NAME, f"{img_name}.png"
            )
            # Determine the landmarks folder based on the image folder
            landmarks_path: str = path_join(
                dirname(dirname(self.path)), LANDMARKS_FOLDER_NAME, f"{img_name}.npy"
            )

        # Verify that the path exists
        # If the image folder does not exists create it
        if not path_exists(dirname(self.path)):
            os.mkdir(dirname(self.path))

        # If the semantic folder does not exists create it
        if not path_exists(dirname(semantic_path)):
            os.mkdir(dirname(semantic_path))

        # If the landmarks folder does not exists create it
        if not path_exists(dirname(landmarks_path)):
            os.mkdir(dirname(landmarks_path))

        # Save the image
        imwrite(self.path, self.image)
        # Save the semantic
        if save_semantic:
            imwrite(semantic_path, self.semantic)

        # If the landmarks have to be saved
        if save_landmarks:
            # Get the landmarks
            landmarks = self.landmark(landmark_model=landmark_model)
            # Save the landmarks
            np_save(landmarks_path, np_array(landmarks))


    def process(
        self, landmark_model: int = LANDMARKS_MEDIAPIPE,
        do_semantic: bool = True, size: Tuple[int, int] = None
    ) -> Face:
        """
        Create the semantic and a square crop of the image

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :param size: Size of the image, defaults to (512, 512)
        :type size: Tuple[int, int], optional
        :raises UserWarning: If the size is not square
        :return: Nothing but do a lot of thing behind ;)
        :rtype: None
        """

        # The processing has to work on non empty images
        assert self.image is not None or self.path is not None

        if self.image is None:
            self.image: np_array = imread(self.path)

        if size is None:
            size = self.image.shape[:2]

        # Set the semantic to 0
        self.semantic = np_zeros(shape=self.image.shape, dtype=int32)

        is_squarred = self.image.shape[0] == self.image.shape[1]
        # Crop the image into a square to avoid modifying the shape of the image
        if not is_squarred:
            self.square_crop(landmark_model)

        # Resize the image to fit the input size of the network which will parse the image
        self.image = resize(self.image, (512, 512))

        # Get the semantic of the image
        if do_semantic:
            self.semantic = self.face_parser.parse(image=self.image).numpy()
            self.semantic = self.semantic.astype(uint8)

        # Resize to get the wanted shape
        if size[0] != size[1]:
            warn.warn(
                "The size entered is not a square size, which will modify the global shape of the image."
            )

        self.resize(size)

        return self


    @classmethod
    def empty(cls, path: str = None) -> Face:
        """
        Create an empty Face object

        :param path: Path to the image file, defaults to None
        :type path: str, optional
        :return: Itself
        :rtype: Face
        """
        if path is None:
            # Create a new path
            path: str = path_join(
                dirname(realpath(__file__)), "images", f"{str(uuid4())}.png"
            )
        # Return a new instance
        return cls(path=path)


    @classmethod
    def from_image(cls, image: np_array) -> Face:
        """
        Create an instance of the class from an image

        :param image: Image to create the object on
        :type image: np_array
        :return: Itself ;)
        :rtype: Face
        """
        # Crate an empty Face Object
        empty = cls.empty()
        empty.image = image
        # Return the empty Face object with the image
        return empty
    

    @classmethod
    def from_tensor(cls, image: torch.Tensor) -> Face:
        """
        Create an instance of the class from an image

        :param image: Image to create the object on
        :type image: np_array
        :return: Itself ;)
        :rtype: Face
        """
        # Crate an empty Face Object
        empty = cls.empty()
        image = (image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1]
        empty.image = image
        # Return the empty Face object with the image
        return empty


    def show(self) -> None:
        """
        Show the image

        :return: None
        :rtype: None
        """
        imshow(self.path, self.image)
        waitKey(0)


    def get_element(self, semantic_value: Union[int, list, str]) -> np_array:
        """
        Show a specified element in the image

        :param semantic_value: The value of the semantic we are searching for
        :type semantic_value: int
        :return: The element wanted
        :rtype: np_array
        """
        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        elif isinstance(semantic_value, str):
            mask = self.semantic == MASK_VALUE
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Get a 3d mask for every color
        mask_colour: np_array = np_array([mask.T, mask.T, mask.T])

        # Every value that is not in the mask will be black
        return self.image * mask_colour.T
    

    def get_element_landamrks(
        self, landmarks_points: Union[List[int], str], landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        mask = self.get_mask_landmarks(landmarks_points=landmarks_points, landmark_model=landmark_model)
        # Get a 3d mask for every color
        mask_colour: np_array = mask[..., None].repeat(3, -1)

        # Every value that is not in the mask will be black
        return self.image * mask_colour


    def crop_over_element(self, semantic_value: Union[int, list]) -> np_array:
        """
        Crop over the wanted semantic element

        :param semantic_value: _description_
        :type semantic_value: Union[int, list]
        :return: _description_
        :rtype: np_array
        """
        if isinstance(semantic_value, str):
            semantic_value = MASK_VALUE[semantic_value]
            return self.crop_over_element(semantic_value)

        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Get the indices that are in the mask
        indices: np_array = np_where(mask > 0)
        crop_min_x, crop_max_x = np_min(indices[0]), np_max(indices[0])
        crop_min_y, crop_max_y = np_min(indices[1]), np_max(indices[1])
        # Get a 3d mask for every color
        mask_colour: np_array = np_array([mask.T, mask.T, mask.T])
        # Get the masked image
        maked_image: np_array = self.image * mask_colour.T
        # Get the image cropped
        cropped_image: np_array = maked_image[crop_min_x:crop_max_x, crop_min_y:crop_max_y, :]

        return cropped_image


    def crop_over_mask(self, semantic_value: Union[int, list]) -> np_array:
        """
        Crop over the wanted semantic element

        :param semantic_value: _description_
        :type semantic_value: Union[int, list]
        :return: _description_
        :rtype: np_array
        """

        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        elif isinstance(semantic_value, np.ndarray):
            mask = semantic_value
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Get the indices that are in the mask
        indices: np_array = np_where(mask > 0)
        crop_min_x, crop_max_x = np_min(indices[0]), np_max(indices[0])
        crop_min_y, crop_max_y = np_min(indices[1]), np_max(indices[1])
        # Get the image cropped
        cropped_mask: np_array = mask[crop_min_x:crop_max_x, crop_min_y:crop_max_y]

        return cropped_mask


    def show_element(
        self, semantic_value: Union[int, list], with_crop: bool = False
    ) -> None:
        """
        Show a semantic element in the image

        :param semantic_value:  The value of the semantic we are searching for
        :type semantic_value: Union[int, list]
        :return: No return
        :rtype: None
        """
        # Show the semantic element
        if not with_crop:
            image_semantic: np_array = self.get_element(semantic_value)
        else:
            image_semantic: np_array = self.crop_over_element(semantic_value)

        # Show the element
        imshow("Semantic value", image_semantic)
        waitKey()


    def show_semantic(self, alpha: float = 0.4) -> None:
        """
        Show the mask over the image

        :param alpha: Transparency of the overlay between 0 and 1, defaults to 0.4
        :type alpha: float, optional
        :return: Nothing
        :rtype: None
        """
        colour_overlay: np_array = np_zeros((*self.size[:2], 3), dtype=uint8)
        colour_overlay[:, :, 0] += self.semantic

        for colour in range(int(np_max(self.semantic))):
            # Get the masks
            mask = colour_overlay[:, :, 0] == colour

            # Change the color
            colour_overlay[mask] = USED_COLOR_OVERLAY[colour]

        imshow("Mask", addWeighted(colour_overlay, alpha, self.image, 1 - alpha, 0))
        waitKey()


    def show_points(self, points: List[Tuple[int, int]], save_path: str = None) -> None:
        """
        Show the landmarks numeroted on the image

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :return: Nothing
        :rtype: None
        """
        # Copy the image
        img_copy = copy(self.image)
        # Write image + all the found cordinate points (x,y)

        scale = min(img_copy.shape[:2]) / 512
        radius = int(2 * scale)

        for k, (x, y) in enumerate(points):
            # write points
            img_copy = circle(img_copy, (int(x), int(y)), radius, (119, 61, 55), -1)
            # write numbers
            img_copy = putText(
                img_copy, text=str(k), org =(int(x) + radius, int(y) + radius),
                color=(119, 61, 55), fontFace=FONT_HERSHEY_SIMPLEX,
                fontScale=0.3 * scale, thickness=1, lineType=cv2.LINE_AA
            )

        img_copy = imutils.resize(img_copy, width=500)

        if save_path is not None:
            imwrite(save_path, img_copy)

        imshow(self.path, img_copy)
        waitKey(0)


    def show_landmarks(
        self, landmark_model: int = LANDMARKS_MEDIAPIPE, chosen: List = None, save_path: str = None
    ) -> None:
        """
        Show the landmarks numeroted on the image

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :return: Nothing
        :rtype: None
        """
        # Get the landmark
        if isinstance(landmark_model, int):
            landmarks: np_array = self.landmark(landmark_model)
        else:
            landmarks = landmark_model

        # Copy the image
        img_copy = copy(self.image)
        # Write image + all the found cordinate points (x,y)

        scale = min(img_copy.shape[:2]) / 512
        radius = int(2 * scale)

        if chosen is None:
            chosen = list(range(0, len(landmarks)))

        for k, (x, y) in enumerate(landmarks[chosen]):
            # write points
            img_copy = circle(img_copy, (int(x), int(y)), radius, (119, 61, 55), -1)
            # write numbers
            img_copy = putText(
                img_copy, text=str(k), org =(int(x) + radius, int(y) + radius),
                color=(119, 61, 55), fontFace=FONT_HERSHEY_SIMPLEX,
                fontScale=0.3 * scale, thickness=1, lineType=cv2.LINE_AA
            )

        img_copy = imutils.resize(img_copy, width=500)

        if save_path is not None:
            imwrite(save_path, img_copy)

        imshow(self.path, img_copy)
        waitKey(0)


    def __getitem__(self, items: Any) -> Union[str, np_array, np_array]:
        """
        Item getter

        :param items: Items index
        :type items: Any
        :return: Items
        :rtype: Union[str, np_array, np_array]
        """
        elements: list = [self.path, self.image, self.semantic]
        return elements[items]
    
    def __str__(self) -> str:
        return f"Face({os.path.splitext(os.path.basename(self.path))[0]})"


    def align(
        self, other_face: Union[List[Face], Face], landmark_model: int = LANDMARKS_MEDIAPIPE,
        landmarks_points: Union[List[int], str, None] = None, self_update: bool = False,
    ) -> Face:
        """
        Align other face to this face

        :param other_face: Face to align
        :type other_face: Union[List[Face], Face]
        :param landmark_model: Landmarks to use for the alignment, defaults to LANDMARKS_MEDIAPIPE
        :type landmark_model: int, optional
        :param landmarks_points: Set of points to use for the alignment, defaults to None
        :type landmarks_points: Union[List[int], str, None], optional
        :param self_update: If True update the given face else only return the aligned face, defaults to False
        :type self_update: bool, optional
        :return: Returns the aligned faces
        :rtype: Face
        """
        
        landmarks = self.landmark(landmark_model)
        if isinstance(landmarks_points, str):
            landmarks_points = LANDMARKS_REGION[landmarks_points]
            landmark_model_align = -1
        elif landmarks_points is None:
            landmarks_points = list(range(len(landmarks)))
            landmark_model_align = landmark_model
        else:
            landmark_model_align = -1

        # If the other face is a single face
        if isinstance(other_face, Face):
            # set it to a list
            other_face = [other_face]

        # Get the faces and landmarks
        faces = [self.image] + [face.image for face in other_face]
        landmarks = [self.landmark(landmark_model)[landmarks_points]] + [face.landmark(landmark_model)[landmarks_points] for face in other_face]

        # Align the faces
        # Skip the first one since it's the reference face, so its the same as the original
        aligned_faces: List[np_array] = align(images=faces, all_points=landmarks, landmark_model=landmark_model_align, skip_ref=True)

        # If we actualy need to update the faces object
        if self_update:
            # Update every faces
            for face, aligned in zip(other_face, aligned_faces):
                # Update the image
                face.image = aligned
                # Uncomment for process aligned image
                # face.process(landmark_model, face.size[::-1])
                face.landmark_cache = {}

        return aligned_faces


    @property
    def has_face(self) -> bool:
        """
        Return True if face is found on the image

        :return: True if face is found on the image else False
        :rtype: bool
        """

        # Convert the image to rgb
        rgbimg: np_array = cvtColor(self.image, COLOR_BGR2RGB)
        rects: np_array = dlib_detector(rgbimg, 1)
        # Return if the number of faces found is above 0
        return len(rects) > 0


    def get_zone_landmarks(
        self,  landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to keep
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size)
        # Create the polly to keep
        mask = fillPoly(mask, pts=[landmarks[landmarks_points]], color=1)

        # Apply the mask
        # First copy the image
        image_copy: np_array = copy(self.image)
        # Apply the mask
        image_copy[mask == 0] = 0

        return image_copy


    def get_squarred_crop_landmarks(
        self,  landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to keep
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        min_x, min_y  = np_min(landmarks[landmarks_points], axis=0)
        max_x, max_y  = np_max(landmarks[landmarks_points], axis=0)

        h, w = max_x - min_x, max_y - min_y
        max_h, max_w = self.image.shape[0] - min_x, self.image.shape[0] - min_y
        side = min([max([h, w]), max_h, max_w])
        side_y, side_x = side, side

        if side == max_h or side == h:
            delta = (side - w) // 2

            margin = 0 if min_y - delta >= 0 else abs(min_y - delta)
            min_y = max([0, min_y - delta])
            
            side_y = side + margin
        elif side == max_w or side == w:
            delta = (side - h) // 2

            margin = 0 if min_x - delta >= 0 else abs(min_x - delta)
            min_x = max([0, min_x - delta])
            
            side_x = side + margin

        # Apply the mask
        # First copy the image
        image_copy: np_array = copy(self.image)
        # Apply the mask
        image_copy = image_copy[min_y:min_y + side_y, min_x:min_x + side_x]

        return image_copy


    def get_out_zone_landmarks(
        self,  landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to keep
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size)
        # Create the polly to keep
        mask = fillPoly(mask, pts=[landmarks[landmarks_points]], color=1)

        # Apply the mask
        # First copy the image
        image_copy: np_array = copy(self.image)
        # Apply the mask
        image_copy[mask != 0] = 0

        return image_copy


    def get_mask_landmarks(
        self, landmarks_points: Union[List[int], str], landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to keep
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size, dtype=uint8)

        if isinstance(landmarks_points, str):
            landmarks_points = LANDMARKS_REGION[landmarks_points]

        if isinstance(landmarks_points[0], list):
            for sub_set in landmarks_points:
                mask += self.get_mask_landmarks(
                    landmarks_points=sub_set, landmark_model=landmark_model
                )
        else: 
            # Create the polly to keep
            mask = fillPoly(mask, pts=[landmarks[landmarks_points]], color=1)

        return mask
    


    def get_mask_landmarks_convex(
        self,  landmarks_points: Union[List[int], str], landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to keep
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size, dtype=uint8)

        if isinstance(landmarks_points, str):
            landmarks_points = LANDMARKS_REGION[landmarks_points]

        if landmarks_points is None:
            landmarks_points = list(range(len(landmarks)))

        if isinstance(landmarks_points[0], list):
            for sub_set in landmarks_points:
                mask += self.get_mask_landmarks_convex(
                    landmarks_points=sub_set, landmark_model=landmark_model
                )
        else: 
            pt = cv2.convexHull(np.array([landmarks[landmarks_points]]).astype(np.int32))[:, 0, :]
            # Create the polly to keep
            mask = fillPoly(mask, pts=[pt], color=1)

        return mask


    def exclude_zone_landmarks(
        self,  landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to remove
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size)
        # Create the polly to keep
        mask = fillPoly(mask, pts=[landmarks[landmarks_points]], color=1)

        # Apply the mask
        # First copy the image
        image_copy: np_array = copy(self.image)
        # Apply the mask
        image_copy[mask == 1] = 0

        return image_copy


    def segmentation_zone_landmarks(
        self,  keep_landmarks_points: List,
        exlude_landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param keep_landmarks_points: List of points to define the area to keep
        :type keep_landmarks_points: List
        :param exlude_landmarks_points: List of points to define the area to keep
        :type exlude_landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size)
        # Create the polly to keep
        mask = fillPoly(mask, pts=[landmarks[keep_landmarks_points]], color=1)
        # Create the polly to remove
        mask = fillPoly(mask, pts=[landmarks[exlude_landmarks_points]], color=0)

        # Apply the mask
        # First copy the image
        image_copy: np_array = copy(self.image)
        # Apply the mask
        image_copy[mask == 0] = 0

        return image_copy


    def get_mask(self, semantic_value: Union[int, list, str]) -> np_array:
        """
        Show a specified element in the image

        :param semantic_value: The value of the semantic we are searching for
        :type semantic_value: int
        :return: The element wanted
        :rtype: np_array
        """

        if isinstance(semantic_value, str):
            semantic_value = MASK_VALUE[semantic_value]

        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Every value that is not in the mask will be black
        return mask.astype("uint8")


    def mask_difference(self, other: Face, threshold: int = 25, closing_iterations: int = 3) -> np_array:
        """
        Get difference between faces

        :param other: _description_
        :type other: Face
        :param threshold: _description_, defaults to 25
        :type threshold: int, optional
        :param closing_iterations: _description_, defaults to 3
        :type closing_iterations: int, optional
        :return: _description_
        :rtype: np_array
        """
        # Get the second face and resize it
        other: Face = deepcopy(other)
        other.resize(self.size)
        difference: np_array = np_abs(self.image - other.image, dtype="int8")
        threshold_array: np_array = np_zeros(difference.shape[:-1], dtype="uint8")
        threshold_array[np_any(difference > threshold, axis=2)] = 1

        prev_threshold_array: np_array = np_zeros(shape=threshold_array.shape)
        while np_any(prev_threshold_array != threshold_array):
            prev_threshold_array: np_array = copy(threshold_array)
            threshold_array: np_array = binary_dilation(input=threshold_array, iterations=3).astype(threshold_array.dtype)
            threshold_array: np_array = binary_erosion(input=threshold_array, iterations=3).astype(threshold_array.dtype)
        
        return threshold_array.astype("uint8")


    def square_pad_crop(
        self, landmarks: List, x_landmarks: List = None, y_landmarks: List = None, 
        landmark_model: int = LANDMARKS_DLIB
    ):
        lm = self.landmark(landmark_model=landmark_model)

        if x_landmarks is not None:
            xlm = lm[x_landmarks][:,1]
        else:
            xlm = np.empty(shape=(0))

        if y_landmarks is not None:
            ylm = lm[y_landmarks][:,0]
        else:
            ylm = np.empty(shape=(0))

        lm = lm[landmarks]

        ymin, ymax = int(np_min(np.hstack((lm[:, 0], ylm)))), int(np_max(np.hstack((lm[:, 0], ylm))))
        xmin, xmax = int(np_min(np.hstack((lm[:, 1], xlm)))), int(np_max(np.hstack((lm[:, 1], xlm))))
        
        img = copy(self.image)[xmin:xmax, ymin:ymax]
        height = xmax - xmin
        width = ymax - ymin

        hpad = max((width - height) // 2, 0)
        wpad = max((height - width) // 2, 0)

        img = cv2.copyMakeBorder(img, hpad, hpad, wpad, wpad, cv2.BORDER_CONSTANT)
        # h, w, c = img.shape

        return img
