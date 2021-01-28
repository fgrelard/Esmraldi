#!/bin/bash
## Script to generate NMF analysis for several imzML files

function print_help {
echo "
Generates analysis of dimension reduction methods
for several imzML files

Usage :
$(basename "$0") -i input_dir -o output_name -t theoretical_spectrum -n number_components

where:
     -i  input root imzML directory (contains 00, 01, 0X subdirectories)
     -o  output (.csv)
     -t  theoretical spectrum (.json)
     -n  number of components for dimension reduction method
     -a  optional switch if alignment is not needed
     -p  optional switch is normalization before NMF is not needed
     -r  optional switch to perform statistical analysis with NMF
     -s  directory number to start with (region);"
}

START_DIR=0
while getopts "i:o:t:n:s:apr" o; do
    case "${o}" in
        i)
            INPUT=${OPTARG}
            ;;
        o)
            OUTPUT=${OPTARG}
            ;;
        t)
            THEORETICAL=${OPTARG}
            ;;
        n)
            NUMBER_COMPONENTS=${OPTARG}
            ;;
        a)
            OPT_NO_ALIGN='true'
            ;;
        p)
            OPT_NORMALIZE='true'
            ;;
        r)
            OPT_ANALYSIS='true'
            ;;
        s)
            START_DIR=${OPTARG}
            ;;
        h|*)
            print_help;
            exit 2
            ;;
        --)
            print_help;
            exit 2
            ;;
    esac
done

DIRECTORY_NAMES=$(find $INPUT -maxdepth 1 -mindepth 1 -type d -print0 | sort -z | xargs -r0 echo)
OUTPUT_NAME=$(basename ${OUTPUT%.*})
OUTPUT_DIR=$(dirname $OUTPUT)
echo $OUTPUT_NAME, $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
for dir in $DIRECTORY_NAMES
do
    name=$(basename $dir)
    imzml=$dir/$name.imzML
    align=$dir/$OUTPUT_NAME.imzML
    nii=$dir/$OUTPUT_NAME.nii
    peaks=$dir/$OUTPUT_NAME.csv
    outdir=$OUTPUT_DIR/$name
    outname=$outdir/$OUTPUT_NAME.csv
    mkdir -p $outdir
    if [[ $name -lt $START_DIR ]];
    then
        continue
    fi
    if [[ ! $OPT_NO_ALIGN ]]; then
        python3 -m examples.spectra_alignment -i $imzml -o $align -p 1000 -n 3 -z 3 -s 0.055 #--theoretical $THEORETICAL --tolerance_theoretical 0.15
        python3 -m examples.tonifti -i $align -o $nii
    fi
    if [[ $OPT_ANALYSIS ]]; then
        if [[ $OPT_NORMALIZE ]]; then
            python3 -m examples.evaluation.analysis_reduction -i $nii -m $peaks -o $outname -n $NUMBER_COMPONENTS -t $THEORETICAL -p -f
        else
            python3 -m examples.evaluation.analysis_reduction -i $nii -m $peaks -o $outname -n $NUMBER_COMPONENTS -t $THEORETICAL -f
        fi
        python3 -m examples.average_images_same_species -i $nii -m $peaks -o $outdir -t $THEORETICAL
    fi
done
