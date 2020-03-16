import matplotlib.pyplot as plt # plotting and showing images
import numpy as np # handling arrays
from skimage.io import imread # reading images
#from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim # structural similarity
import scipy.ndimage as nd
from scipy.ndimage.filters import uniform_filter

import matplotlib.pyplot as plt  # plotting
from skimage.io import imread  # read in images
import numpy as np  # linear algebra / matrices
import scipy as sc
import scipy.ndimage as nd





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




mse = lambda img1, img2: np.sum(np.power(img1-img2,2))

d=np.mean(imread('data/testpattern.png'),2)

scales = [0.1, 0.5, 1, 10, 20, 100];

#signal to noise
SNR = (np.mean(d))/(np.std(d))

Ntests = 10;

# initialize arrays for results
mse1 = np.zeros((len(scales), Ntests), dtype = np.float32)
ssim1 = np.zeros((len(scales), Ntests), dtype = np.float32)

#nd.median_filter(test, size=4)
from scipy.ndimage.filters import uniform_filter
current_filter = lambda img: anisodiff(img,100,30,0.1,(1.,1.),0.0,1,False)

for i, c_scale in enumerate(scales):
    for j in range(Ntests):
        x = current_filter(d+c_scale*np.random.uniform(-c_scale, c_scale, size = d.shape))
        mse1[i,j]=mse(d,x);
        ssim1[i,j]=ssim(d,x);


    # Add some lines here to display the latest image in a subplot

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Anisotropic Diffusion Filter', fontsize=16)
ax1.loglog(scales,np.mean(mse1,1)) # Add annotations for the plot and axes
ax1.set_title('MSE vs Scale')
ax1.set_xlabel('Scale')
ax1.set_ylabel('MSE')

ax2.semilogx(scales,np.mean(ssim1,1))
ax2.set_title('SSIM vs Scale')
ax2.set_xlabel('Scale')
ax2.set_ylabel('SSIM')

plt.show()





print(mse1)
print("----")
print(ssim1)