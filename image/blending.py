import cv2
import numpy as np


def fix(backgroud, src_pos, trgt_pos, size, margin=5):
    src_roi = backgroud[
        src_pos[0] - int(size[0] // 2) - margin:src_pos[0] + int(size[0] // 2) + margin,
        src_pos[1] - int(size[1] // 2) - margin:src_pos[1] + int(size[1] // 2) + margin
    ]
    src_mask = cv2.GaussianBlur(np.ones(src_roi.shape, src_roi.dtype) * 255, (5, 5), 0, 0) # pylint: disable=E1101

    return cv2.seamlessClone(src_roi, backgroud, src_mask, trgt_pos, cv2.NORMAL_CLONE) # pylint: disable=E1101
