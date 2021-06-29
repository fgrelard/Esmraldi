import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage.draw as draw
import SimpleITK as sitk
import esmraldi.imageutils as utils
import esmraldi.registration as reg
import skimage.transform as transform
import cv2 as cv

def create_data():
    img = np.zeros((100, 100))
    rr, cc = draw.rectangle((15, 33), (85,66), shape=img.shape)
    img[rr, cc] = 255

    rr, cc = draw.ellipse(50, 33, 35, 7)
    img[rr, cc] = 0

    # rr, cc = draw.ellipse(50, 66, 35, 7)
    # img[rr, cc] = 0

    return img

def create_u(factor, radius, rfunc=[], nb_points=100):
    if not rfunc:
        rfunc = [0 for i in range(nb_points)]
    img = np.zeros((100, 100))
    x = np.linspace(-factor*np.pi, factor*np.pi, nb_points)
    y = 2*np.cosh(x/factor)
    xtrans = x + 50
    ytrans = y + 35
    for i, (xc, yc) in enumerate(zip(xtrans, ytrans)):
        current_rfunc = rfunc[i]
        r = radius + current_rfunc
        rr, cc = draw.disk((xc, yc), r, shape=img.shape)
        img[rr, cc] = 255

    return img

nb_points = 100
r = 9
m = 5.0
rfunc = [m - (1.0/(nb_points//(2*m)))*np.abs(nb_points//2 - float(i)) for i in range(nb_points)]
rfunc = [(i//50) * (m - (1.0/(nb_points//(2*m)))*float(i)) for i in range(nb_points)]
reference = create_u(7, r)
target = create_u(7, r, rfunc)


points = np.array([[50, 12], [51, 13], [52, 12], [80, 12]])

local_radius = utils.local_radius(target)

reference = transform.rotate(reference, 180)
reference = transform.resize(reference, (50, 50), order=0)
reference = np.pad(reference, 25)

reference_itk = sitk.GetImageFromArray(reference)
target_itk = sitk.GetImageFromArray(target)

reference_local_max = utils.radius_maximal_balls(reference_itk)
target_local_max = utils.radius_maximal_balls(target_itk)



# fig, ax = plt.subplots(1, 2)
# ax[1].imshow(target)
# ax[0].imshow(reference)
# plt.show()


dt_reference = utils.compute_DT(reference_itk)
dt_target = utils.compute_DT(target_itk)


normalized_dt_reference = np.zeros_like(reference)
np.divide(sitk.GetArrayFromImage(dt_reference), reference_local_max, out=normalized_dt_reference, where=reference_local_max!=0)
normalized_dt_target = np.zeros_like(target)
np.divide(sitk.GetArrayFromImage(dt_target), target_local_max, out=normalized_dt_target, where=target_local_max!=0)

normalized_dt_reference = normalized_dt_reference**10
normalized_dt_target = normalized_dt_target**10

normalized_dt_reference_itk = sitk.GetImageFromArray(normalized_dt_reference)
normalized_dt_target_itk = sitk.GetImageFromArray(normalized_dt_target)

normalized_dt_reference_itk = utils.local_max_dt(reference_itk)
normalized_dt_target_itk = utils.local_max_dt(target_itk)



# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(normalized_dt_reference)
# ax[1].imshow(normalized_dt_target)
# plt.show()


R = reg.register(reference_itk, target_itk, 15, 0.1, find_best_rotation=True, learning_rate=0.00001, update_DT=False, normalize_DT=False)
out = R.Execute(dt_target)

R_update = reg.register(normalized_dt_reference_itk, normalized_dt_target_itk, 15, 0.1, find_best_rotation=True, learning_rate=0.00001, update_DT=False, normalize_DT=True)
out_norm = R_update.Execute(normalized_dt_target_itk)
out_update = R_update.Execute(dt_target)

mse_original = utils.mse(normalized_dt_reference_itk, normalized_dt_target_itk)
mse_after = utils.mse(normalized_dt_reference_itk, out_norm)
print("original=", mse_original, "after=", mse_after)


diff = sitk.GetArrayFromImage(out) - sitk.GetArrayFromImage(out_update)
R_parameters = R.GetTransform().GetParameters()
R_update_parameters = R_update.GetTransform().GetParameters()

print("No update : scale=", R_parameters[0], ", angle=", R_parameters[1])
print("With update : scale=", R_update_parameters[0], ", angle=", R_update_parameters[1])

fig, ax = plt.subplots(1,5)
ax[0].imshow(sitk.GetArrayFromImage(dt_reference))
ax[1].imshow(sitk.GetArrayFromImage(dt_target))
ax[2].imshow(sitk.GetArrayFromImage(out))
ax[3].imshow(sitk.GetArrayFromImage(out_update))
ax[4].imshow(diff)
plt.show()
