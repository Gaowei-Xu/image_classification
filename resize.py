import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import pickle
import cv2
from cStringIO import StringIO
import PIL
import caffe
import time



img_raw = cv2.imread('./test.JPG', cv2.IMREAD_COLOR)
img = cv2.resize(img_raw, (640, 480))
cv2.imwrite('./demo.jpg', img)

