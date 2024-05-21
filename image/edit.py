import cv2
from face.facealign import LANDMARKS_DLIB, LANDMARKS_MEDIAPIPE_FACE2POINTS
from face.faceclass.face import Face, LANDMARKS_MEDIAPIPE
from image.filtre.gradient import gradient
import numpy as np
import scipy
from image.blending import fix
import sys
from tqdm import tqdm
from typing import Union, List
import PIL
import os


PATCHES_PT = [
    [186, 216, 206, 92],
    [216, 207, 205, 206],
    [207, 187, 50, 205],
    [123, 50, 187, 147],
    [123, 147, 137, 227],
    [205, 50, 118, 101],
    [410, 322, 426, 436],
    [436, 426, 425, 427],
    [427, 411, 280, 425],
    [280, 352, 376, 411],
    [352, 376, 366, 401],
    [425, 330, 347, 280],
    [201, 200, 199, 208],
    [200, 424, 428, 199]
]


LANDMARKS_REGION = {
    "LionWrinkle": [107, 108, 151, 337, 336, 285, 413, 168, 189, 55],
    "LionWrinkleWithGlasses": [107, 108, 151, 337, 8],
    "UpperLipWrinkle": [40, 92, 165, 167, 164, 393, 394, 322, 270, 269, 267, 0, 37, 39],
    "ForeheadWrinkle": [70, 71, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 301, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63],
    "MarionetteWrinkle": [
        [409, 410, 436, 434, 364, 365, 379, 378, 395, 431, 424, 335, 321, 375, 291], 
        [185, 186, 216, 214, 135, 136, 150, 149, 170, 211, 204, 106, 91, 146, 61]
    ],
    "CrowsFootWrinkle": [
        [359, 342, 353, 383, 368, 264, 447, 345, 340, 261, 255],
        [430, 113, 124, 156, 139, 34, 227, 116, 11, 31, 25],

    ],
    "UnderEyeWrinkle": [
        [463, 341, 256, 252, 253, 254, 339, 255, 261, 340, 345, 352, 280, 330, 329, 277, 343, 412, 465, 464],
        [243, 112, 26, 22, 23, 24, 110, 25, 31, 111, 116, 123, 50, 101, 100, 47, 114, 188, 245, 244]
    ],
    "NasobialWrinkle": [
        [358, 266, 425, 427, 434, 432, 287, 410, 322, 391, 327],
        [129, 36, 205, 207, 214, 212, 57, 186, 92, 165, 98]
    ],
}


MASK_VALUE = {
    "Lips": [12, 13],
    "Hair": [17]
}


def get_hist(img, exclude_zero=True):
    if img.ndim == 3:
        return np.swapaxes(np.stack([
            get_hist(c[..., 0], exclude_zero=exclude_zero) 
            for c in np.split(img, img.shape[-1], axis=-1)
        ]), 0, 1)
    
    hist, _ = np.histogram(img.flatten(), bins=255, range=(0, 255))
    if exclude_zero:
        hist[0] = 0

    total = np.sum(hist)
    return np.cumsum(hist / total)


def transform_table(hist1, hist2):
    assert hist1.shape == hist2.shape

    if hist1.ndim == 2:
        return np.swapaxes(np.stack([
            transform_table(ch1[..., 0], ch2[..., 0]) 
            for ch1, ch2 in zip(
                np.split(hist1, hist1.shape[-1], axis=-1), 
                np.split(hist2, hist2.shape[-1], axis=-1)
            )
        ]), 0, 1)
    
    table = np.zeros_like(hist1).astype(np.uint16)
    mem_start = 0
    for idx1, vh1 in enumerate(hist1):
        for idx2, vh2 in enumerate(hist2[mem_start:]):
            if vh2 >= vh1:
                table[idx1] = idx2 + mem_start
                mem_start = idx2 + mem_start
                break
        else:
            table[idx1] = idx2

    return table


def color_transfer(img, table):
    if img.ndim == 3:
        assert table.ndim == 2
        return np.stack([color_transfer(c[..., 0], ctable[..., 0]) for c, ctable in zip(np.split(img, img.shape[-1], axis=-1), np.split(table, table.shape[-1], axis=-1))], axis=-1)
    
    img_aligned = np.zeros_like(img)
    for idx, value in enumerate(table):
        img_aligned[img == idx] = value

    return img_aligned


def histogram_match(img1, img2, exclude_zero=True):
    hist1, hist2 = get_hist(img1, exclude_zero), get_hist(img2, exclude_zero)
    table = transform_table(hist1, hist2)
    return color_transfer(img1, table)


def transfer_color(source, target, elements):

    if isinstance(elements, str):
        elements = MASK_VALUE[elements]

    if isinstance(source, str):
        f1 = Face(source).read_image()
        f1.process()
    else:
        f1 = Face(None, source)
        f1.process()

    mask1 = f1.get_mask(elements)[..., None].repeat(3, axis=-1).astype(bool)
    img1 = f1.image * mask1

    f2 = Face(target).read_image()
    f2.process()
    mask2 = f2.get_mask(elements)[..., None].repeat(3, axis=-1).astype(bool)
    img2 = f2.image * mask2

    img1_aligned = histogram_match(img1, img2)
    cv2.imwrite("./PTE_Results/matched.png", img1_aligned)

    img1 = img1_aligned * mask1 + f1.image * ~mask1

    return img1


def wrinkle_remover(path: str, landmarks: Union[str, List[int]] = "LionWrinkle"):
    ldmk_name = landmarks
    if isinstance(landmarks, str):
        landmarks = LANDMARKS_REGION[landmarks]

    use_face2point = isinstance(ldmk_name, str) and ldmk_name == "ForeHeadWrinkle"

    img = cv2.imread(path)
    f = Face.from_image(img)
    # f.show_landmarks(LANDMARKS_MEDIAPIPE, None, "./ldmk.png")

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    grad_s = gradient(s).astype(np.float64)
    grad_v = gradient(v).astype(np.float64)

    grad_v /= np.max(grad_v)
    grad_s /= np.max(grad_s)

    mem_wrinkles = grad_s
    # cv2.imwrite("./gradient.png", (mem_wrinkles * 255).astype(np.uint8))

    masks = f.get_mask_landmarks(landmarks, landmark_model=LANDMARKS_MEDIAPIPE_FACE2POINTS if use_face2point else LANDMARKS_MEDIAPIPE)
    name, ext = os.path.splitext(os.path.basename(path))
    cv2.imwrite(os.path.join(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\These\TheseUtils\PTE_Results\PGT_exemple", f"{name}_{ldmk_name}_mask{ext}"), masks * 255)

    grad_v *= masks
    grad_s *= masks

    grad_v = (grad_v * 255).astype(np.uint8)[masks == 1]
    grad_s = (grad_s * 255).astype(np.uint8)[masks == 1]

    _, grad_v = cv2.threshold(grad_v, 127.5, 255, cv2.THRESH_OTSU)
    _, grad_s = cv2.threshold(grad_s, 127.5, 255, cv2.THRESH_OTSU)
    wrinkles = np.zeros(img.shape[:2], np.uint8)
    wrinkles[masks == 1] = (grad_s[:,0]==255) * 255

    # wrinkles = scipy.ndimage.binary_closing(wrinkle, iterations=3).astype(np.uint8) * 255
    # cv2.imwrite("./gradient_lion.png", wrinkles)

    contours, hierarchy = cv2.findContours(wrinkles, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.array([
        cv2.contourArea(c) for c in contours
    ])
    q75, q50, q25 = np.percentile(areas[areas > 0], [75, 50, 25])
    detected_wrinkles = np.where(areas > (q25))[0]
    # cv2.imwrite("./wrinkle.png", cv2.drawContours(img, tuple([contours[i] for i in detected_wrinkles]), -1, (255, 255, 255), -1))

    # selected_wrinkles = cv2.drawContours(np.zeros(img.shape), [contours[cidx] for cidx in detected_wrinkles], -1, (255, 255, 255), -1)
    # cv2.imwrite("./wrinkle_lion.png", selected_wrinkles)

    # Find healthy skin
    min_std = sys.maxsize
    target = None
    for points in PATCHES_PT:
        tmp_mask = f.get_mask_landmarks(points, landmark_model=LANDMARKS_MEDIAPIPE)
        nonzeros = np.nonzero(tmp_mask)
        min_x, max_x, min_y, max_y = np.min(nonzeros[0]), np.max(nonzeros[0]), np.min(nonzeros[1]), np.max(nonzeros[1])

        tmp_mask = tmp_mask[min_x:max_x, min_y:max_y]
        tmp_wrinkles = mem_wrinkles[min_x:max_x, min_y:max_y][tmp_mask]

        if min_std > np.std(tmp_wrinkles):
            ldmk = f.landmark(LANDMARKS_MEDIAPIPE)[points]
            target = [int((np.max(ldmk[:, 1]) + np.min(ldmk[:, 1])) // 2), int((np.max(ldmk[:, 0]) + np.min(ldmk[:, 0])) // 2)]

    for i in detected_wrinkles:
        wrinkle = contours[i].squeeze(1)
        size = (np.max(wrinkle[:,1]) - np.min(wrinkle[:,1]), np.max(wrinkle[:,0]) - np.min(wrinkle[:,0]))
        img = fix(img, target, (int((np.min(wrinkle[:,0]) + np.max(wrinkle[:,0])) // 2), int((np.min(wrinkle[:,1]) + np.max(wrinkle[:,1])) // 2)), size)

    return img


def stylegan_crop(
    read_path: str, write_path: str, output_size: int = 1024, 
    transform_size: int = 1024, enable_padding: bool = True, 
    random_shift: float = 0.0
):
    img = cv2.imread(read_path)
    img_scale_factor = img.shape[0] / img.shape[1]
    img = cv2.resize(img, (1500, int(1500 * img_scale_factor)))

    face = Face.from_image(img)
    img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    try:
        lm = face.landmark(landmark_model=LANDMARKS_DLIB)
    except RuntimeError:
        return

    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c0 = eye_avg + eye_to_mouth * 0.1

    # Load in-the-wild image.
    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2

    # Keep drawing new random crop offsets until we find one that is contained in the image
    # and does not require padding
    for _ in range(100):
        # Offset the crop rectange center by a random shift proportional to image dimension
        # and the requested standard deviation
        c = (c0 + np.hypot(*x)*2 * random_shift * np.random.normal(0, 1, c0.shape))
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        if not (crop[0] < 0 or crop[1] < 0 or crop[2] >= img.width or crop[3] >= img.height):
            # We're happy with this crop (either it fits within the image, or retries are disabled)
            break

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    # Save aligned image.
    img.save(write_path)
