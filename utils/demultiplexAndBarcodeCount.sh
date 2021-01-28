#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o demultiplexAndBarcodeCount-%A_%a.out
#SBATCH -e demultiplexAndBarcodeCount-%A_%a.err
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL

ml anaconda3/4.1.1
unset PYTHONPATH
source activate bclab

# Given FASTQ files, demultiplexes samples by P1 adapters and then count barcodes in each file
if [ "$#" -ne 4 ]
then
    echo "Usage: demultiplexAndBarcodeCount.sh fastqToSample.txt spacerSeq bcSize path/to/this/script/"
    echo "\t- fastqToSample.txt is formatted as [fastq file name] [batch name (e.g. library1)] [p1ToSample.txt] [barcodeToSequence.txt]
    \tWhere p1ToSample.txt is formatted as [p1 sequence] [sample name (e.g. Rna1)]
    \t and barcodeToSequence.txt is a file that contains barcodes in the first field and CRS identifiers in the second field."
    echo "\t- spacerSeq is the sequence that is between the p1 adapter and cBC."
    echo "\t- bcSize is the expected size of the cBC"
    exit 1
fi

fastqToSample=$1
spacerSeq=$2
bcSize=$3
scriptPath=$4

# If SLURM_ARRAY_TASK_ID is not set, that means either (1) this script is being run locally or (2) there is no job array. In this case, it is assumed that there is only one FASTQ file to demultiplex
if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
    SLURM_ARRAY_TASK_ID=1
fi

line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" "${fastqToSample}")
fastq=$(echo "${line}" | cut -f1)
name=$(echo "${line}" | cut -f2)
p1ToSample=$(echo "${line}" | cut -f3)
bcToCrsFile=$(echo "${line}" | cut -f4)

sh "${scriptPath}"/demultiplexFastq.sh "${fastq}" "${p1ToSample}" "${spacerSeq}" "${bcSize}" "${name}" "${scriptPath}"

# For each sample in the original FASTQ file, count barcodes.
while read p1Line
do
    p1=$(echo "${p1Line}" | cut -f1)
    sample=$(echo "${p1Line}" | cut -f2)
    prefix="${name}${sample}"
    p1AndSpacer="${p1}${spacerSeq}"

    sh "${scriptPath}"/fastqCountBarcodes.sh "${prefix}" "${p1AndSpacer}" "${bcSize}" "${bcToCrsFile}" "${scriptPath}"

done < "${p1ToSample}"
