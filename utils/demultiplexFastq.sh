#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o demultiplexFastq-%A_%a.out
#SBATCH -e demultiplexFastq-%A_%a.err
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL

ml anaconda3/4.1.1
unset PYTHONPATH
source activate bclab

# A wrapper for demultiplexing a FASTQ file.
if [ "$#" -ne 6 ]
then
    echo "Usage: demultiplexFastq.sh my.fastq p1ToSample.txt spacerSeq bcSize outPrefix path/to/this/script"
    echo "\tFor a description of these arguments, see the Python script"
fi

fastq=$1
p1ToSample=$2
spacer=$3
bcSize=$4
outPrefix=$5
scriptPath=$6

echo "Demultiplexing ${outPrefix}"
python3 "${scriptPath}"/demultiplex_fastq.py --prefix "${outPrefix}" "${fastq}" "${p1ToSample}" "${spacer}" "${bcSize}"
