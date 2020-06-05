import numpy as np
import matplotlib.pyplot as plt
import esmraldi.imzmlio as imzmlio
import esmraldi.segmentation as seg
import esmraldi.fusion as fusion
import SimpleITK as sitk
import scipy.spatial.distance as distance
import argparse

"""
Computes the cosine distance between each MALDI ion image and the MRI image
And sorts the MALDI according to this distance in descending order
"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-m", "--mri", help="Input MRI image (ITK format)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold", default=0)
parser.add_argument("-r", "--ratio", help="Compute ratio images", action="store_true")


args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
threshold = int(args.threshold)
is_ratio = args.ratio


if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    image = imzmlio.to_image_array(imzml)
    mzs, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(inputname)).T
    mzs = [i for i in range(image.shape[2])]
    mzs = np.asarray(mzs)

print("Mass-to-charge ratio=", mzs)

image = image[..., mzs >= threshold]
image = imzmlio.normalize(image)

mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

image_mri = sitk.GetArrayFromImage(sitk.ReadImage(mriname, sitk.sitkUInt8)).T

if is_ratio:
    ratio_images, ratio_mzs = fusion.extract_ratio_images(image, mzs)
    image = np.concatenate((image, ratio_images), axis=2)
    mzs = np.concatenate((mzs, ratio_mzs))

image_flatten = seg.preprocess_pca(image).T
image_mri = imzmlio.normalize(image_mri)
image_mri_flatten = seg.preprocess_pca(image_mri)


fig, ax = plt.subplots(1, 2)
ax[0].imshow(image_mri)
ax[1].imshow(image[..., 0])
plt.show()

distances = []
for i in range(image_flatten.shape[-1]):
    maldi = image_flatten[..., i]
    print(np.amax(image_mri_flatten))
    d = distance.cosine(maldi, image_mri_flatten)
    distances.append(d)

indices = [i for i in range(len(distances))]
indices.sort(key=lambda x:distances[x], reverse=False)

indices_array = np.array(indices)
print(distances)
print(indices_array)

similar_images = np.take(image, indices, axis=-1)
similar_mzs = np.take(mzs, indices)

print(similar_mzs)

plt.imshow(similar_images[..., 0])
plt.show()
