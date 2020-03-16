import matplotlib.pyplot as plt  # plotting
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



#fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,5))
#ax1.imshow(d)
#ax2.imshow(test)
#ax3.imshow(test2)

#plt.show()


#-#


def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0.0,option=1,ploton=False):
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import matplotlib.pyplot as plt
        from time import sleep

        plt.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        plt.subplot(1, 3, 1)
        plt.imshow(img, interpolation='nearest')
        plt.title('Original')
        plt.colorbar()

    for ii in np.arange(0, niter):
        smoothimgout = imgout

        if sigma != 0:
            smoothimgout = imgout  ###### Introduce gradient smoothing here
            smoothimgout = nd.gaussian_filter(diffused, 10, 0)

        # calculate the diffs
        deltaS[:-1, :] = np.diff(smoothimgout, axis=0)
        deltaE[:, :-1] = np.diff(smoothimgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    if ploton:
        iterstring = "Iteration %i" % (ii + 1)
        plt.subplot(1, 3, 2)
        plt.imshow(imgout)
        plt.title(iterstring)

        plt.subplot(1, 3, 3)
        plt.imshow(img - imgout)
        plt.title('Difference before - after')

    return imgout

test_out = anisodiff(diffused,10,50,0.1,(1.,1.),0.0,1,False)

test_out_100 = anisodiff(diffused,1000,50,0.1,(1.,1.),0.0,1,False)

test_out_cond1 = anisodiff(diffused,10,1,0.1,(1.,1.),0.0,1,False)

test_out_cond30 = anisodiff(diffused,1000,30,0.1,(1.,1.),0.0,1,False)

test_out_cond100 = anisodiff(diffused,1000,100,0.1,(1.,1.),0.0,1,False)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,5))
ax1.imshow(diffused)
ax2.imshow(test_out)
ax3.imshow(test_out_100)

ax4.imshow(test_out_cond1)
ax5.imshow(test_out_cond30)
ax6.imshow(test_out_cond100)

plt.show()

