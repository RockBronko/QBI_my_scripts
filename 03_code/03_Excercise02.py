

import time
import numpy as np
import pandas as dp
import skimage.io as io
import matplotlib.pyplot as plt
#from keras_datasets import mnist

from sklearn.neighbors import KNeighborsClassifier

img = io.imread('../03-files/ct_tiles.tif')
labels = dp.read_csv('../03-files/malignancy.csv')

sample_size = 10
dimensions = np.shape(img)
image_array = np.arange(0,dimensions[0],int(dimensions[0]/(sample_size-1)))
images = img[image_array]

import keras_preprocessing.image

img_aug = keras_preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=05.0,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.25,
    zoom_range=0.5,
    fill_mode='nearest',
    horizontal_flip=False,
    vertical_flip=False
)

tiles = np.expand_dims(img[image_array], -1)
tileLabels = labels.malignancy[image_array]

# setup augmentation
img_aug.fit(tiles)

#fig, ori = plt.subplots(1, 10, figsize=(13, 3))
  #  for originals in ori:
 #       next(originals.imshow(images[i]))

fig, (series1) = plt.subplots(1,  3, figsize=(13, 3))
k = 0
j = 10
i = 0
aug = []
lab = []
while 10 >= k:
    real_aug = img_aug.flow(tiles[:10], tileLabels[:10], batch_size=10, shuffle=False)

    augmented, augmented_label = next(real_aug)
    aug[i:j] = augmented
    lab[i:j] = augmented_label

  #  for series2, aug2 in zip(series1, augmented):
  #      print(np.shape(aug))
  #      series2.imshow(aug2[:, :, 0])
    i = i+10
    j = j+10
    k = k + 1
print(lab)


#Nearest neighbor
neigh_class = KNeighborsClassifier(n_neighbors=1)
neigh_class.fit(augmented.reshape((sample_size, -1)), augmented_label)

neigh_class.predict(augmented.reshape((sample_size, -1)))

#confusion matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Stolen from: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return ax1


pred_values = neigh_class.predict(tiles.reshape((sample_size, -1)))
ax1 = print_confusion_matrix(confusion_matrix(tileLabels, pred_values), class_names=range(2))
ax1.set_title('Nearest Neighbour(Real Data) Accuracy: {:2.2%}'.format(accuracy_score(tileLabels, pred_values)));
#plt.show()
