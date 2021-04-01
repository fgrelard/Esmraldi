"""
Module for the annotation of a spectrum
"""

import numpy as np
from collections.abc import Iterable
import numbers

def closest_peak(reference_mz, theoretical_spectrum, tolerance):
    """
    Closest peak for a given m/z ratio,
    from a theoretical spectrum.

    Parameters
    ----------
    reference_mz: float
        peak m/z
    theoretical_spectrum: dict
        theoretical spectrum, where keys are m/z ratio and values the names
    tolerance: float
        acceptable mz delta to consider peaks are equal

    Returns
    ----------
    list
        peaks in theoretical spectrum closest to reference_mz

    """
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
    """
    Annotate an observed spectrum by comparison
    to a theoretical spectrum.

    Parameters
    ----------
    observed: list
        mass list
    theoretical: dict
        theoretical spectrum generated from several species rule
    tolerance: float
        acceptable mz delta to consider peaks are equal

    Returns
    ----------
    dict
        annotated mass list
    """
    annotated = {}
    for peak in observed:
        closest_peaks = closest_peak(peak, theoretical, tolerance)
        annotated[peak] = closest_peaks
    return annotated


def annotation_ratio(observed, theoretical, tolerance=0.1):
    """
    Annotate an observed spectrum by comparison
    to a theoretical spectrum. The observed spectrum
    may contain ratio between two species.

    Parameters
    ----------
    observed: list
        mass list
    theoretical: dict
        theoretical spectrum generated from several species rule
    tolerance: float
        acceptable mz delta to consider peaks are equal

    Returns
    ----------
    dict
        annotated mass list
    """
    O = []
    OT = set()
    for o in observed:
        if isinstance(o, numbers.Number):
            O += [o]
        elif o is not None:
            OT |= {i for i in o}
    annotated = annotation(O, theoretical, tolerance)
    annotated_tuples = annotation(list(OT), theoretical, tolerance)
    ratios = [o for o in observed if o not in O]
    for peak in ratios:
        value = [annotated_tuples[peak[0]], annotated_tuples[peak[1]]]
        annotated[peak] = value
    order_annotated = {k:annotated[k] for k in observed}
    return order_annotated
