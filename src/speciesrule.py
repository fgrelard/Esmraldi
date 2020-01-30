import json
import numpy as np
from functools import partial

def parts_eval(i, parts):
    current_name = ""
    for part in parts:
        part_split = part.split("+")
        is_i = any([item=="i" for item in part_split])
        if is_i:
            current_name += str(eval(part))
        else:
            current_name += part
    return current_name


def json_to_species(filename):
    species = []

    with open(filename) as f:
        data = json.load(f)

    rules = data["rules"]
    fns = []
    for rule in rules:
        name = rule["name"]
        category = rule["category"]
        mz = rule["mz"]
        count = rule["count"] if "count" in rule else 1
        begin = rule["begin"] if "begin" in rule else None
        end = rule["end"] if "end" in rule else None
        naming_fn = rule["naming_fn"] if "naming_fn" in rule else None
        add_fn = rule["adduct_fn"] if "adduct_fn" in rule else None
        if naming_fn is not None:
            parts = naming_fn.split(",")
            naming_fn = lambda i, parts=parts: parts_eval(i, parts)
            fns.append(naming_fn)
        else:
            naming_fn = lambda i: name + str(i)

        s = SpeciesRule(name=name, category=category, mz=mz, count=count, begin=begin, end=end, naming_fn=naming_fn, adduct_fn=add_fn)
        species.append(s)
    return species


class SpeciesRule:
    def __init__(self, name, category, mz, count=1, begin=None, end=None, naming_fn=None, adduct_fn=None):
        self.name = name
        self.category = category
        self.mz = mz

        if begin is not None:
            self.begin = begin
        else:
            self.begin = mz

        if begin is not None and end is not None:
            self.count = int((end - begin) // self.mz) + 1
        else:
            self.count = count

        if naming_fn is None:
            self.naming_fn = lambda i: self.name + str(i)
        else:
            self.naming_fn = naming_fn

        self.adduct_fn = adduct_fn


    def species(self):
        d = {}
        for i in range(self.count):
            current_name = self.naming_fn(i + 1)
            current_mz = self.begin + i * self.mz
            d[current_name] = current_mz
        return d
