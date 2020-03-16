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

d=imread('data/scroll.tif')

print(d.shape)
print(np.mean(d))

test = d + (10000 * np.random.randn(*d.shape))
test2 = d + (10000 * np.random.randn(*d.shape))

med4= nd.median_filter(test, size=4)
med8= nd.median_filter(test, size=8)
med12= nd.median_filter(test, size=12)

#built in iterations
iterations = 10

sun4 = nd.uniform_filter(test, size=4)
sun8 = nd.uniform_filter(test, size=4)
sun12 = nd.uniform_filter(test, size=4)

un4 = nd.uniform_filter(test, size=4)
un8 = nd.uniform_filter(test, size=4)
un12 = nd.uniform_filter(test, size=4)


for i in range (1,iterations):
    un4 = nd.uniform_filter(un4, size=4)
    un8 = nd.uniform_filter(un8, size=4)
    un12 = nd.uniform_filter(un12, size=4)


fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(15,5))
ax1.imshow(test)
ax1.set_title('noised_image')
ax2.imshow(sun4)
ax2.set_title('4by4 kernel')
ax3.imshow(sun8)
ax3.set_title('8by8 kernel')
ax4.imshow(sun12)
ax4.set_title('12by12 kernel')
ax5.imshow(d)
ax5.set_title('original')
ax6.imshow(test)
ax7.imshow(un4)
ax8.imshow(un8)
ax9.imshow(un12)
ax10.imshow(d)

plt.show()
