import numpy as np
import matplotlib.pyplot as plt
import similaritymeasures
import scipy.ndimage
import src.spectraprocessing as sp
import scipy.signal as signal

def distance_two_distributions(dis1, dis2):
    sum = 0
    n1 = len(dis1)
    for i in range(n1):
        val1 = dis1[i]
        val2 = dis2[i]
        distance = abs(val2 - val1)
        sum += distance / max(val1, val2)
    return sum / n1


def weighted_hdistance(dis1, dis2, index, index2):
    factor = abs(index2 - index)/max(len(dis1), len(dis2))
    dist = abs(maldi_curvature[index] - mri_curvature[index_closest])
    return dist * factor

def find_peaks(data, prominence, w):
    peaks, _ = signal.find_peaks(tuple(data),
                                 prominence=prominence,
                                 wlen=w,
                                 distance=3)
    return peaks



mri_curvature = np.loadtxt("data/mri_curvature.txt")
maldi_curvature = np.loadtxt("data/registered_curvature.txt")

#Shift distribution
index_max_mri = mri_curvature.argmax()
index_max_maldi = maldi_curvature.argmax()
mri_curvature = np.roll(mri_curvature, -index_max_mri)
maldi_curvature = np.roll(maldi_curvature, -index_max_maldi)

#Resize distribution
mri_curvature = scipy.ndimage.zoom(mri_curvature, len(maldi_curvature)/len(mri_curvature), order=3)
maldi_curvature = scipy.ndimage.gaussian_filter1d(np.copy(maldi_curvature), 7)
mri_curvature = scipy.ndimage.gaussian_filter1d(np.copy(mri_curvature), 7)

indices_mri = [0]
indices_mri += (find_peaks(mri_curvature, 0.01, 30)).tolist()

indices_maldi = [0]
indices_maldi += (find_peaks(maldi_curvature, 0.01, 30)).tolist()
print(indices_maldi)
print(indices_mri)
search_size = 40
realigned_maldi_curvature = np.copy(maldi_curvature)
MAX = 2**32
indices_to_closest = {key:MAX for key in indices_mri}
indices_to_width = {key:10 for key in indices_maldi}

# Extract closest peak from MALDI to MRI in horizontal distance
l = []
for index in indices_mri:
    val = sp.closest_peak(index, indices_to_width)
    l.append(abs(val[0] - index))

#Average and stddev of horizontal distance
print("Average h-distance=", np.mean(l), " stddev=", np.std(l))

plt.plot(mri_curvature, "b", label="Courbure IRM")
plt.plot(maldi_curvature, "g", label="Courbure MALDI")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#Average vertical relative distance
d = distance_two_distributions(maldi_curvature, mri_curvature)
d2 = distance_two_distributions(mri_curvature, maldi_curvature)
print("Average v-distance =", d, " ", d2)
