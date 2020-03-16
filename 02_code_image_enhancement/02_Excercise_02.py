import matplotlib.pyplot as plt  # plotting
from skimage.io import imread  # read in images
import numpy as np  # linear algebra / matrices
# make the notebook interactive
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets  # add new widgets
from IPython.display import display

from numpy.random import randn


d=np.mean(imread('data/testpattern.png'),2)

x1 = 5
x2 = 30
y1 = 5
y2 = 200

subD1=d[x1:x2,y1:y2];
snrD1=np.mean(subD1)/np.std(subD1)

print(snrD1)


d_snr100=d+scale_100*randn(*d.shape);

#d_snr10 = d + scale_10 * randn(*d.shape);


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(d, cmap = 'gray')
ax2.imshow(d_snr100, cmap = 'gray')
ax3.imshow(d_snr10, cmap = 'gray')
plt.show()