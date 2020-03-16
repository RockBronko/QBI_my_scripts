
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
