
"""import numpy as np


def normalize_numpy(img, mean, denominator):
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def normalize(img, mean, std, max_pixel_value=255.0):

    #modified to make it possible to enter 
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        return normalize_cv2(img, mean, denominator)
    return normalize_numpy(img, mean, denominator)"""