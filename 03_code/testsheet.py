import numpy as np
import pandas as dp
import skimage.io as io
import matplotlib.pyplot as plt

img = io.imread('03-Files\ct_tiles.tif')
labels = dp.read_csv('03-Files\malignancy.csv')

print('Image size: {0},{1},{2}'.format(img.shape[0],img.shape[1],img.shape[2]))
print('Labels {0}'.format(labels.malignancy.count() ))

from keras_preprocessing.image import ImageDataGenerator
img_aug = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=30.0,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.5,
    fill_mode='nearest',
    horizontal_flip=False,
    vertical_flip=False
)

import numpy as np
import matplotlib.pyplot as plt

tiles = np.expand_dims(img[0:6500:450], -1)
tileLabels = labels.malignancy[0:6500:450]
fig, m_axs = plt.subplots(4, 10, figsize=(14, 9))
# setup augmentation
img_aug.fit(tiles)
real_aug = img_aug.flow(tiles[:20], tileLabels[:20], batch_size=100, shuffle=False)
for c_axs, do_augmentation in zip(m_axs, [False, True, True, True]):
    print("aha")
    if do_augmentation:
        img_batch, label_batch = next(real_aug)
    else:
        img_batch, label_batch = tiles, tileLabels
    for c_ax, c_img, c_label in zip(c_axs, img_batch, label_batch):
        print(np.shape(img_batch))
        c_ax.imshow(c_img[:, :, 0], cmap='viridis', vmin=0, vmax=255)
        c_ax.set_title('{}\n{}'.format(
            c_label, 'aug' if do_augmentation else ''))
        c_ax.axis('off')

plt.show()