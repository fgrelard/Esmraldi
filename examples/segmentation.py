import src.segmentation as seg
import src.imzmlio as imzmlio
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from skimage import segmentation

from skimage.filters import gaussian
from skimage.morphology import binary_erosion

image = nib.load("/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel.nii")

img_data = image.get_data()
img_data = np.pad(img_data, (3,3), 'constant')


similar_images = seg.find_similar_images(img_data)


print(len(similar_images))
# for k in range(similar_images.shape[-1]):
#     print(k)


fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gaussian(current_img.T,1), cmap=plt.cm.gray)
ax.plot(edges[:,0], edges[:,1], '.')
ax.plot(snake_curve[:, 0], snake_curve[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, current_img.shape[1], current_img.shape[0], 0])

plt.show()



#imzmlio.to_nifti(img_data[..., y_kmeans==2], "/mnt/d/MALDI/imzML/MSI_20190419_01/00/clustered_k3_c2.nii")
