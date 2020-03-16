
import matplotlib.pyplot as plt # plotting
from skimage.io import imread # read in images
import numpy as np # linear algebra / matrices
# make the notebook interactive
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets #add new widgets
from IPython.display import display
class idict(dict):
    def __init__(self,*args,**kwargs) : dict.__init__(self,*args,**kwargs)
    def __str__(self): return 'ImageDictionary'
    def __repr__(self): return 'ImageDictionary'


a=imread('data/scroll.tif')
b=imread('data/wood.tif')
c=imread('data/asphalt_gray.tif')

#matplotlib inline
# setup the plotting environment
#plt.imshow(a, cmap = 'gray'); # show a single image

x1 = 10
x2 = 30
y1 = 10
y2 = 420


subB1=b[x1:x2,y1:y2];
snrB1=np.mean(subB1)/np.std(subB1) # compute the snr
print("SNR for B_1 is {}".format(snrB1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.imshow(b, cmap = 'gray')
ax2.imshow(subB1, cmap = 'gray')
plt.show()
