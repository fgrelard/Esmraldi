import csv
import argparse
import numpy as np
import xlsxwriter

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--observed", help="Observed annotation (.csv)")

parser.add_argument("-t", "--theoretical", help="Theoretical annotation (.csv)")
parser.add_argument("-o", "--output", help="Output file")
args = parser.parse_args()

def to_dict(annotation):
    masses = {}
    for row in annotation:
        k = float(row[0])
        v = [row[2+i] for i in range(len(row)-2)]
        masses[k] = v
    return masses

def missing_masses(theoretical, observed, tol=0.05):
    masses_theoretical = np.array(list(theoretical.keys()))
    masses_observed = np.array(list(observed.keys()))
    out = masses_theoretical[(np.abs(masses_observed[:, None] - masses_theoretical) >= tol).all(0)]
    return out

def missing_annotation(theoretical, observed, tol=0.05):
    masses_th = np.array(list(theoretical.keys()))
    masses_obs = np.array(list(observed.keys()))
    union_th = masses_th[(np.abs(masses_obs[:, None] - masses_th) < tol).any(0)]
    union_obs = masses_obs[(np.abs(masses_th[:, None] - masses_obs) < tol).any(0)]
    names_th = [[t for t in theoretical[k] if t != ""] for k in union_th]
    names_obs = [[o for o in observed[k] if o != ""] for k in union_obs]


    names_th_flatten = np.unique([elem for l in names_th for elem in l]).tolist()
    names_obs_flatten = np.unique([elem for l in names_obs for elem in l]).tolist()
    missing = [elem for elem in names_obs_flatten if elem not in names_th_flatten]

    missing_in_th = {}
    for i in range(len(names_th)):
        current_th = names_th[i]
        current_obs = names_obs[i]
        for elem in current_obs:
            if elem not in current_th:
                if i not in missing_in_th:
                    missing_in_th[union_th[i]] = []
                if i == 4:
                    print(elem)
                missing_in_th[union_th[i]].append(elem)
    return missing_in_th



theoretical_name = args.theoretical
observed_name = args.observed
output_name = args.output

with open(theoretical_name, "r") as f:
    theoretical = list(csv.reader(f, delimiter=";"))

with open(observed_name, "r") as f:
    observed = list(csv.reader(f, delimiter=";"))


nb_theoretical = len(theoretical)
nb_observed = len(observed)

full_theoretical = to_dict(theoretical)
full_observed = to_dict(observed)

annotated_theoretical = {k:v for k, v in full_theoretical.items() if any([elem != "" for elem in v])}
annotated_observed = {k:v for k, v in full_observed.items() if any([elem != "" for elem in v])}

nb_annotated_theoretical = len(annotated_theoretical)
nb_annotated_observed = len(annotated_observed)

masses_th = np.array(list(annotated_theoretical.keys()))
masses_obs = np.array(list(annotated_observed.keys()))
union_th = masses_th[(np.abs(masses_obs[:, None] - masses_th) < 0.05).any(0)]
union_obs = masses_obs[(np.abs(masses_th[:, None] - masses_obs) < 0.05).any(0)]
names_th = [[t for t in annotated_theoretical[k] if t != ""] for k in union_th]
names_obs = [[o for o in annotated_observed[k] if o != ""] for k in union_obs]

names_th_flatten = np.unique([elem for l in names_th for elem in l]).tolist()
names_obs_flatten = np.unique([elem for l in names_obs for elem in l]).tolist()
union_names = [elem for elem in names_obs_flatten if elem in names_th_flatten]
missing_th_flatten = [elem for elem in names_obs_flatten if elem not in names_th_flatten]
missing_obs_flatten = [elem for elem in names_th_flatten if elem not in names_obs_flatten]

missing_full_theoretical = missing_masses(full_observed, full_theoretical)
missing_full_observed = missing_masses(full_theoretical, full_observed)
missing_theoretical = missing_masses(annotated_observed, annotated_theoretical)
missing_observed = missing_masses(annotated_theoretical, annotated_observed)
missing_annotation_theoretical = missing_annotation(annotated_theoretical, annotated_observed)
missing_annotation_observed = missing_annotation(annotated_observed, annotated_theoretical)
print(missing_annotation_theoretical)

stats_detection_th = [len(missing_full_theoretical), len(theoretical) - len(missing_full_observed), nb_theoretical]
stats_detection_obs = [len(missing_full_observed), len(observed) - len(missing_full_theoretical),  nb_observed]


workbook = xlsxwriter.Workbook(output_name)
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

worksheet = workbook.add_worksheet("Peak detection")
worksheet2 = workbook.add_worksheet("Annotation (global)")
worksheet3 = workbook.add_worksheet("Annotation (peak by peak)")

#Worksheet 1
headers = ["Missing", "Common", "Total"]
for i in range(len(headers)):
    worksheet.write(0, i+1, headers[i], header_format)

worksheet.write(1, 0, "Manual", header_format)
worksheet.write(2, 0, "Computed", header_format)
worksheet.write_row(1, 1, stats_detection_th)
worksheet.write_row(2, 1, stats_detection_obs)
worksheet.write(0, 6, "Missing masses in computed", header_format)
worksheet.write_column(1, 6, missing_full_observed)
worksheet.write_column(1, 7, [", ".join([name for name in full_theoretical[k] if name != ""]) for k in missing_full_observed])

#Worksheet 2
for i in range(len(headers)):
    worksheet2.write(0, i+1, headers[i], header_format)

worksheet2.write(1, 0, "Manual", header_format)
worksheet2.write(2, 0, "Computed", header_format)
worksheet2.write_row(1, 1, [len(missing_theoretical), nb_annotated_theoretical - len(missing_observed), nb_annotated_theoretical])
worksheet2.write_row(2, 1, [len(missing_observed), nb_annotated_observed - len(missing_theoretical), nb_annotated_observed])

worksheet2.write(0, 6, "Missing masses in computed", header_format)
worksheet2.write_column(1, 6, missing_observed)
worksheet2.write_column(1, 7, [", ".join([name for name in full_theoretical[k] if name != ""]) for k in missing_observed])

worksheet2.write(0, 9, "Missing masses in manual", header_format)
worksheet2.write_column(1, 9, missing_theoretical)
worksheet2.write_column(1, 10, [", ".join([name for name in full_observed[k] if name != ""]) for k in missing_theoretical])

#Worksheet 3
for i in range(len(headers)):
    worksheet3.write(0, i+1, headers[i], header_format)

worksheet3.write(1, 0, "Manual", header_format)
worksheet3.write(2, 0, "Computed", header_format)

worksheet3.write_row(1, 1, [len(missing_th_flatten), len(union_names), len(names_th_flatten)])
worksheet3.write_row(2, 1, [len(missing_obs_flatten), len(union_names), len(names_obs_flatten)])

worksheet3.write(0, 6, "Missing masses in computed", header_format)
worksheet3.write_column(1, 6, missing_annotation_observed)
worksheet3.write_column(1, 7, [", ".join([elem for elem in l]) for l in list(missing_annotation_observed.values()) ])

worksheet3.write(0, 9, "Missing masses in manual", header_format)
worksheet3.write_column(1, 9, missing_annotation_theoretical)
worksheet3.write_column(1, 10, [", ".join([elem for elem in l])  for l in list(missing_annotation_theoretical.values()) ])


workbook.close()
# print(len(missing_annotation_theoretical))
# print(len(missing_annotation_observed))

# print(missing_annotation_theoretical)
# print(missing_annotation_observed)


# print([str(i) + ":" + "".join(annotated_theoretical[i]) for i in missing_theoretical])

# print([str(i) + ":" + ",".join([v for v in annotated_observed[i] if v != ""]) for i in missing_observed])
# print(missing_theoretical.shape, missing_observed.shape)
# print(nb_theoretical, nb_annotated_theoretical, nb_observed, nb_annotated_observed)

#print(observed)
