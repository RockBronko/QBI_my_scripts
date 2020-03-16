import matplotlib.pyplot as plt  # plotting
from numpy.core.multiarray import ndarray
from skimage.io import imread  # read in images
import numpy as np  # linear algebra / matrices
import scipy as sc
import scipy.ndimage as nd


# make the notebook interactive
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets  # add new widgets
from IPython.display import display

from numpy.random import randn

d=imread('data/testpattern.tif')
diffused = d + (40 * np.random.randn(*d.shape)+ 127)
diffused = (255 / np.amax(diffused))*diffused
imgout = nd.gaussian_filter(diffused, 10, 0)

fig, (one, two, three) = plt.subplots(1,3, figsize=(15,5))

one.imshow(d)
two.imshow(diffused)
three.imshow(imgout)

plt.show()