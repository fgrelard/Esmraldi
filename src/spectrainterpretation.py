import numpy as np

def closest_peak(reference_mz, theoretical_spectrum, tolerance):
    keys = list(theoretical_spectrum.keys())
    values = list(theoretical_spectrum.values())
    closest = -1
    closest_name = None
    closest_peaks = {}

    for v in values:
        diff = abs(reference_mz - v)
        if diff < tolerance or np.isclose(diff, tolerance):
            closest = v
            closest_names = [k for k,v in theoretical_spectrum.items() if np.isclose(v, closest)]
            for closest_name in closest_names:
                closest_peaks[closest_name] = diff

    closest_peaks = {k: v for k, v in sorted(closest_peaks.items(), key=lambda item: item[1])}
    list_closest = list(closest_peaks.keys())
    return list_closest

def annotation(observed, theoretical, tolerance=0.1):
    annotated = {}
    for peak in observed:
        closest_peaks = closest_peak(peak, theoretical, tolerance)
        annotated[peak] = closest_peaks
    return annotated
