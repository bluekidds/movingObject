import cv2
import numpy as np

def isMoving(current, fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(), threshold = 10000):
    '''
    Input : current frame
    Return: Binary if sum is greater than threshold
    '''
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    return fgmask.sum() > threshold
