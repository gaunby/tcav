
import matplotlib.image as mpimg
import PIL
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf

filename = '/work3/s174498/ImageNet_Data/random500_5/tn_sl2.jpg'

image = PIL.Image.open(filename)
width, height = image.size

print('Image Height       : ',height)
print('Image Width        : ',width)

shape = (299, 299)
print(np.array(image))
img = np.array(
          PIL.Image.open(tf.io.gfile.GFile(filename,
                                           'rb')).convert('RGB').resize(
                                               shape, PIL.Image.BILINEAR),
          dtype=np.float32)
print(img)
