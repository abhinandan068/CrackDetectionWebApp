import cv2
import numpy as np


def preprocess_clf(img):
    img = cv2.resize(img, (160, 160))
    img = img / 255.0
    return np.reshape(img, (1, 160, 160, 3))


def preprocess_seg(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.reshape(img, (1, 128, 128, 3))
