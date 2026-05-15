import numpy as np


def calculate_severity(mask):

    crack_pixels = np.sum(mask > 0.5)

    total_pixels = mask.size

    ratio = crack_pixels / total_pixels

    if ratio < 0.01:
        return "Low", ratio

    elif ratio < 0.03:
        return "Medium", ratio

    else:
        return "High", ratio
