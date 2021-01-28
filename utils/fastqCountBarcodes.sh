#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o fastqCountBarcodes-%A_%a.out
#SBATCH -e fastqCountBarcodes-%A_%a.err
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL

ml anaconda3/4.1.1
unset PYTHONPATH
source activate bclab

# Given a FASTQ file, count barcodes. This script can be called by another script with 5 arguments or used directly as a SLURM wrapper.

# Calling from another script
if [ "$#" -eq 5 ]
then
    prefix=$1
    p1AndSpacer=$2
    bcSize=$3
    bcToCrsFile=$4
    scriptPath=$5
# Calling from SLRUM
elif [ ! -z "${SLURM_ARRAY_TASK_ID}" ]
then
    argfile=$1
    scriptPath=$2
    line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" "${argfile}")
    prefix=$(echo "${line}" | cut -f1)
    p1AndSpacer=$(echo "${line}" | cut -f2)
    bcSize=$(echo "${line}" | cut -f3)
    bcToCrsFile=$(echo "${line}" | cut -f4)
# Else error
else
   echo "Usage: fastqCountBarcodes.sh fastqPrefix p1AndSpacer bcSize bcToCrs.txt path/to/this/script/"
   echo "\t- fastqPrefix is both the prefix for a file called fastqPrefix.fastq and the prefix for output files."
   echo "\t- p1AndSpacer is the concatenation of the p1 adapter and the spacer before the barcode."
   echo "\t- bcSize is the expected barcode size."
   echo "\t- bcToCrs.txt is a file that contains barcodes in the first field and CRS identifiers in the second field."
   echo "\nOR"
   echo "sbatch --array=1-arraySize fastqCountBarcodes.sh argumentFile.txt path/to/this/script/"
   echo "\tWhere argumentFile.txt has 4 fields per line, corresponding to fastqPrefix, p1AndSpacer, bcSize, bcToCrs.txt"
   echo "\tWhen called this way, the script runs as a SLURM job array."
   exit 1
fi

# Pull lines from the FASTQ that contain at least 30 consecutive letters in the set [ACGTN] ...
grep -e "[ACGTN]\{30,\}" "${prefix}".fastq |
# ...then get the ${bcSize} [ACGT] letters immediately after the p1 and spacer, i.e. the cBC...
sed -n 's/.*'"${p1AndSpacer}"'\([ACGT]\{'"${bcSize}"'\}\).*/\1/p' |
# ...sort and count the barcodes...
sort | uniq -c |
# ...reformat to get rid of leading spacer and change the delimiter to a tab
sed -e 's/^ *//;s/ /\t/' > "${prefix}".tmp

python3 "${scriptPath}"/fastq_count_barcodes.py "${prefix}" "${bcToCrsFile}"

# Clean up
rm "${prefix}".tmp
