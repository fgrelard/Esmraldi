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
        end_mz = rule["end_mz"] if "end_mz" in rule else None
        family_number = rule["family_number"] if "family_number" in rule else None
        count_per_mol = rule["count_per_mol"] if "count_per_mol" in rule else count
        naming_fn = rule["naming_fn"] if "naming_fn" in rule else None
        add_fn = rule["adduct_fn"] if "adduct_fn" in rule else None
        if naming_fn is not None:
            parts = naming_fn.split(",")
            naming_fn = lambda i, parts=parts: parts_eval(i, parts)
            fns.append(naming_fn)
        else:
            naming_fn = lambda i: name + str(i)

        s = SpeciesRule(name=name, category=category, mz=mz, count=count, count_per_mol=count_per_mol, begin=begin, end_mz=end_mz, family_number=family_number, naming_fn=naming_fn, adduct_fn=add_fn)
        species.append(s)
    return species


class SpeciesRule:
    def __init__(self, name, category, mz, count=1, count_per_mol=1, begin=None, end_mz=None, family_number=None, naming_fn=None, adduct_fn=None):
        self.name = name
        self.category = category
        self.mz = mz

        if begin is not None:
            self.begin = begin
        else:
            self.begin = 1

        if end_mz is not None:
            self.count = int(end_mz // self.mz - self.begin + 1)
        else:
            self.count = count

        self.count_per_mol = count_per_mol

        if family_number is not None:
            self.family_number = family_number
        else:
            self.family_number = 1

        if naming_fn is None:
            self.naming_fn = lambda i: str(i) + self.name
        else:
            self.naming_fn = naming_fn

        if adduct_fn is None:
            self.adduct_fn = ".*"
        else:
            self.adduct_fn = adduct_fn


    def species(self):
        d = {}
        for i in range(self.count):
            index = i + self.begin
            current_name = self.naming_fn(index) if index > 0 else ""
            current_mz = index * self.mz
            d[current_name] = current_mz
        return d
