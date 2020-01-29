import numpy as np

def closest_peak(reference_mz, theoretical_spectrum, tolerance):
    keys = list(theoretical_spectrum.keys())
    values = list(theoretical_spectrum.values())
    closest = -1
    closest_name = None
    for v in values:
        diff = abs(reference_mz - v)
        if diff < tolerance or np.isclose(diff, tolerance):
            closest = v
            closest_name = keys[values.index(closest)]
    return closest_name

def annotation(observed, theoretical, tolerance=0.1):
    annotated = {}
    for peak in observed:
        closest_name = closest_peak(peak, theoretical, tolerance)
        annotated[peak] = closest_name
    return annotated
