import numpy as np
import os

for root, dirs, files in os.walk("/mnt/d/Stats/BLE/250/50/results/"):
    for file in files:
        if file.endswith(".csv"):
            basename = os.path.splitext(file)[0]
            array = np.loadtxt(os.path.join(root, file))
            np.savetxt(os.path.join(root, basename+"_v2.csv"), array, delimiter=",", fmt="%10.5f")
