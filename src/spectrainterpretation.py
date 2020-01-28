import numpy as np
import re

def update_spectrum(original, new):
    for key in new.keys():
        if key not in original:
            original[key] = new[key]
    return original

def molecules_from_rule(n, mol_mz, naming_fn, first_index=0):
    d = {}
    for i in range(first_index, n):
        current_name = naming_fn(i)
        current_mz = mol_mz + i * mol_mz
        d[current_name] = current_mz
    return d

def molecules_from_range(start_mz, end_mz, spacing_mz, naming_fn):
    n = int((end_mz - start_mz) // spacing_mz)
    return molecules_from_rule(n, start_mz, naming_fn)

def molecule_adducts(mol_name, mol_mz, adducts):
    d = {}
    for name, mz in adducts.items():
        current_mz = mol_mz + mz
        current_name = mol_name + "_" + name
        d[current_name] = current_mz
    return d

def molecule_adducts_regexp(references, expression, adducts):
    theoretical = {}
    pattern = re.compile(expression)
    list_names = '\n'.join(list(references.keys()))
    matches = pattern.findall(list_names)
    mol_add = {k:references[k] for k in matches if k in references}
    for name, mz in mol_add.items():
        ion_with_adducts = molecule_adducts(name, mz, adducts)
        theoretical.update(ion_with_adducts)
    return theoretical

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
