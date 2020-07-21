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
     -p  optional switch is normalization before NMF is not needed";
}

while getopts "i:o:t:n:ap" o; do
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
            OPT_NO_NORMALIZE='true'
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
    if [[ ! $OPT_NO_ALIGN ]]; then
        python3 -m examples.spectra_alignment -i $imzml -o $align -p 250 -n 3 -z 2 -s 0.055
        python3 -m examples.tonifti -i $align -o $nii
    fi
    if [[ ! $OPT_NO_NORMALIZE ]]; then
        python3 -m examples.evaluation.analysis_reduction -i $nii -m $peaks -o $outname -n $NUMBER_COMPONENTS -t $THEORETICAL -p
    else
        python3 -m examples.evaluation.analysis_reduction -i $nii -m $peaks -o $outname -n $NUMBER_COMPONENTS -t $THEORETICAL
    fi
    python3 -m examples.average_images_same_species -i $nii -m $peaks -o $outdir -t $THEORETICAL
done
