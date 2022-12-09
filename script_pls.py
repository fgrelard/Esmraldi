import argparse
import os
import sys
import subprocess

def extract_parent_paths(names):
    paths = []
    for imzml_file in names:
        parent_path = os.path.abspath(os.path.join(imzml_file, os.pardir))
        parent_path = parent_path.replace(" ", "\\ ")
        mask_path = "--regions " + parent_path + os.path.sep + "masks/resized/*.tif"
        paths.append(mask_path)
    return paths

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Training", action="store_true")
parser.add_argument("--test", help="Testing", action="store_true")
parser.add_argument("--validation", help="Validation", action="store_true")
parser.add_argument("--bootstrap", help="Do Bootstrap", action="store_true")
parser.add_argument("--lasso", help="Switch to LASSO", action="store_true")
parser.add_argument("--parameters_train", help="Nb components or alpha", nargs="+", type=float)
args = parser.parse_args()

is_train = args.train
is_test = args.test
is_lasso = args.lasso
is_validation = args.validation
is_bootstrap = args.bootstrap
parameters_train = args.parameters_train
print(parameters_train)

test_datasets = {}
home_folder = "/home/fgrelard/Data/Vaclav/"

test_datasets["P2D3"] = home_folder + "CROPPED_20220801 Pratt 2-D3 DHB/20220801_168x603_5um_Pratt2-D3_DHBspray_POS_mode_67-1000mz_70K_Laser35_5KV_350C_Slens90_peakpicked3_cropped.imzML"
test_datasets["P2F4"] = home_folder + "20220801 Pratt 2-F4 DHB/20220802_116x132_5um_Pratt2-F4_DHBspray_POS_mode_50-750mz_70K_Laser35_5KV_350C_Slens90_peak_picked.imzML"
test_datasets["P2D6"] = home_folder + "20220802 Pratt2 - D6 DHB 5um/20220802_141x83_5um_Pratt2-D6_DHBspray_POS_mode_50-750mz_70K_Laser35_5KV_350C_Slens90_peakpicked.imzML"
test_datasets["P3E5"] = home_folder + "20220805 Pratt 3-E5 DHB/20220804_242x209_5um_Pratt3 E5 #3_DHBspray_POS_mode_55-820mz_70K_Laser35_4P5KV_350C_Slens90_peakpicked.imzML"
test_datasets["P6F1"] = home_folder + "20220805 Pratt 6-F1 DHB/20220805_131x154_5um_Pratt6 F1 #3_DHBspray_POS_mode_55-820mz_70K_Laser35_4P5KV_350C_Slens90_peak_picked.imzML"
test_datasets["P7E4"] = home_folder + "20221104 Pratt7E4 #2 - 12um DHB/20220920_146x196_5um_Pratt7 E4 #2_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned500.imzML"
test_datasets["P2C4"] = home_folder + "20221108 Pratt2C4 #4  - 12um DHB/20221108_176x313_5um_Pratt2C4 #4_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned500.imzML"
test_datasets["P2A4"] = home_folder + "20221122 Pratt2A4  #3 - DHB 5um/20221121_144x428_5um_Pratt2A4 #3_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned250.imzML"
test_datasets["P6E1"] = home_folder + "20221108 Pratt6E1 #5 - 12um DHB/20221107_192x520_5um_Pratt6E1 #5_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned500.imzML"
test_datasets["P6B5"] = home_folder + "20221110 Pratt6B5 #3 - DHB 5um/20221109_150x387_5um_Pratt6B5 #3_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned250.imzML"

test_datasets["P7D2"] = home_folder + "20221108 Pratt7D2 #3 - DHB 5um/20221108_108x262_5um_Pratt7D2 #3_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned100.imzML"
test_datasets["P7C5"] = home_folder + "20221020 Pratt7-C5 - DHB/20221019_216x256_5um_P7C5_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned100.imzML"
test_datasets["P2D5Rot"] = home_folder + "20221118 Pratt2D5 DHB - RotSprayer/20221024_163x261_5um_P2D5_DHBRotSpray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned250.imzML"
test_datasets["P2D5TM"] = home_folder + "20221118 Pratt2D5 DHB - TMSprayer/20221007_157x431_5um_P2D5_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned500.imzML"
test_datasets["P3E3"] = home_folder + "20220803 Pratt 3-E3 DHB/20220804_121x92_5um_Pratt3 E3 #2_DHBspray_POS_mode_55-820mz_70K_Laser35_4P5KV_350C_Slens90_aligned75.imzML"

test_datasets["SBD3"] = home_folder + "20220922 SBD3 - DHB/20220922_184x516_5um_SBD3_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_peakpicked.imzML"
test_datasets["SBD5"] = home_folder + "20221107 SBD5 #2 - 12um DHB/20221105_164x266_5um_SBD5 #2_DHBspray_POS_mode_60-900mz_70K_Laser35_4P5KV_350C_Slens90_aligned150.imzML"
test_datasets["DiPaolo"] = home_folder + "20220804 DiPaolo DHB/20220803_422x363_5um_Di Paolo Area 4#10_DHBspray_POS_mode_55-820mz_70K_Laser35_4P5KV_350C_Slens90_aligned500.imzML"

train_keys = ["P6E1", "P2D3", "P2F4", "P7E4", "P6B5", "P2C4", "P2A4"]
validation_keys = ["P6F1", "P2F4", "P2D3"]

train = {key: test_datasets[key] for key in train_keys}
validation = {key: test_datasets[key] for key in validation_keys}
name_files = "_".join(train_keys)

training_dir = home_folder + "trainingsets" + os.path.sep
name_dir = training_dir + name_files + os.path.sep

model_dir = home_folder + "models" + os.path.sep
outmodel_dir = model_dir + name_files + os.path.sep

validation_dir = home_folder + "validation" + os.path.sep
outvalidation_dir = validation_dir + name_files + os.path.sep

name_file = "train.tif"
validation_file = "validation.tif"
method = "lasso" if is_lasso else "pls"
name_models = [model_dir + name_files + os.path.sep + "model_" + method + "_" + str(i) + ".joblib" for i in parameters_train]
# regions = ["Casein", "Collagen", "ET", "LO", "Matrix"]
regions = ["*"]
prefix_region_names = name_dir + "regions/"
region_names = [prefix_region_names + r + ".tif " for r in regions]


bootstrap_repetitions = 5
if is_train:
    names = [s.replace(" ", "\\ ") for s in train.values()]
    name_input = " ".join(names)
    parent_paths = extract_parent_paths(train.values())
    masks_paths = " ".join(parent_paths)
    output_set = name_dir + name_file
    for i in range(bootstrap_repetitions):
        if not os.path.exists(name_dir):
            os.makedirs(name_dir, exist_ok=True)
            currdir = name_dir + str(i) + os.path.sep
            os.makedirs(currdir, exist_ok=True)
            currset = currdir + name_file
            cmd = "python3 -m create_image_for_pls -i " + name_input + " " + masks_paths + " -o " + currset + " --sample_size 1000"
            subprocess.call(cmd, shell=True)
        for j, name in enumerate(name_models):
            currname = os.path.splitext(name)[0] + "_" + str(i) + ".joblib"
            currregions = [name_dir + str(i) +os.path.sep+ "regions" + os.path.sep + os.path.basename(r) for r in region_names]
            cmd_train = "python3 -m examples.pls -i " + currset + " -r "+ "".join(currregions) + " -o " + currname
            if is_lasso:
                cmd_train += " --lasso --alpha " + str(parameters_train[j])
            else:
                cmd_train += " --nb_component " + str(parameters_train[j])
            subprocess.call(cmd_train, shell=True)

if is_bootstrap:
    for i, name_model in enumerate(name_models):
        name_model_file = os.path.splitext(os.path.basename(name_model))[0]
        name_model_dir = os.path.splitext(name_model)[0]
        bootstrap_names = [name_model_dir + "_" + str(i) + os.path.sep + name_model_file + "_" + str(i) + ".joblib" for i in range(bootstrap_repetitions)]
        out_bootstrap =  name_model_dir + os.path.sep + name_model_file + ".joblib"
        cmd_bootstrap = "python3 -m examples.bootstrap_model -i " + " ".join(bootstrap_names) + " -o " + out_bootstrap
        print(out_bootstrap)
        if is_lasso:
            cmd_bootstrap += " --lasso"
        os.makedirs(name_model_dir)
        subprocess.call(cmd_bootstrap, shell=True)

if is_validation:
    validation_dir = validation_dir + name_files + os.path.sep
    validation_names = [s.replace(" ", "\\ ") for s in validation.values()]
    validation_input = " ".join(validation_names)
    parent_paths = extract_parent_paths(validation.values())
    validation_masks_paths = " ".join(parent_paths)
    validation_output = outvalidation_dir + name_file
    print(validation_output, validation_masks_paths)
    if not os.path.exists(outvalidation_dir):
        os.makedirs(outvalidation_dir, exist_ok=True)
        cmd = "python3 -m create_image_for_pls -i " + validation_input + " " + validation_masks_paths + " -o " + validation_output + " --sample_size 1000"
        print(cmd)
        subprocess.call(cmd, shell=True)
    cmd_validation = "python3 -m examples.evaluate_models -i " + outmodel_dir + " --validation_dataset " + validation_output
    print(cmd_validation)
    if is_lasso:
        cmd_validation += " --lasso"
    subprocess.call(cmd_validation, shell=True)

if is_test:
    for i, name_model in enumerate(name_models):
        name_model_file = os.path.basename(name_model)
        name_model_dir = os.path.splitext(name_model)[0] + os.path.sep
        input_model = name_model_dir + name_model_file
        os.makedirs(name_model_dir + "results/", exist_ok=True)
        for key, name_test in test_datasets.items():
            name_test_escape = name_test.replace(" ", "\\ ")

            out_test_dir = name_model_dir + "results/" + key + ".png"
            cmd = "python3 -m examples.pls_test -i " + input_model + " -t " + name_test_escape + " -o " + out_test_dir
            print(cmd)
            subprocess.call(cmd, shell=True)